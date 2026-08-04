from unittest.mock import Mock, patch

import io
import json

from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from rest_framework import status
from rest_framework.test import APITestCase

from app.tenants.models import MembershipStatusChoices, Team, TeamMembership, TeamMembershipRoleChoices, Tenant
from app.user.models import User


@override_settings(DATABASE_ROUTERS=[])
class G3InferenceViewTest(APITestCase):
    def setUp(self):
        cache.clear()
        tenant = Tenant.objects.create(name="G3 Tenant", slug="g3-tenant")
        self.team = Team.objects.create(tenant=tenant, name="G3 Team", slug="g3-team")
        self.user = User.objects.create_user(email="g3-admin@test.com", password="pass")
        TeamMembership.objects.create(
            tenant=tenant,
            team=self.team,
            user=self.user,
            role=TeamMembershipRoleChoices.TEAM_ADMIN,
            status=MembershipStatusChoices.ACTIVE,
        )
        self.client.force_authenticate(self.user)

    @override_settings(G3_INFERENCE_URL="https://g3.example", G3_INFERENCE_TOKEN="secret")
    @patch("app.g3_inference.views.requests.post")
    def test_admin_can_run_inference(self, post):
        post.return_value = Mock(status_code=200)
        post.return_value.json.return_value = {"drag_coefficient": 0.31, "preview_png": "data:image/png;base64,x"}
        response = self.client.post(
            "/v1/g3/inferences",
            {"stl": SimpleUploadedFile("car.stl", b"solid car\nendsolid car", content_type="model/stl")},
            format="multipart",
            HTTP_X_TEAM_ID=str(self.team.id),
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["drag_coefficient"], 0.31)
        self.assertEqual(post.call_args.kwargs["headers"], {"Authorization": "Bearer secret"})

    @override_settings(
        G3_INFERENCE_URL="",
        G3_INFERENCE_TOKEN="",
        AWS_S3_BUCKET="private-config-bucket",
    )
    @patch("app.g3_inference.views.requests.post")
    @patch("app.g3_inference.views.boto3.client")
    def test_service_config_can_be_loaded_from_private_s3(self, boto_client, post):
        boto_client.return_value.get_object.return_value = {
            "Body": io.BytesIO(json.dumps({"url": "https://g3.example", "token": "s3-secret"}).encode())
        }
        post.return_value = Mock(status_code=200)
        post.return_value.json.return_value = {"drag_coefficient": 0.31}

        response = self.client.post(
            "/v1/g3/inferences",
            {"stl": SimpleUploadedFile("car.stl", b"solid car\nendsolid car")},
            format="multipart",
            HTTP_X_TEAM_ID=str(self.team.id),
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        boto_client.return_value.get_object.assert_called_once_with(
            Bucket="private-config-bucket",
            Key="_private/g3/inference-service.json",
        )
        self.assertEqual(post.call_args.kwargs["headers"], {"Authorization": "Bearer s3-secret"})

    @override_settings(G3_INFERENCE_URL="https://g3.example", G3_INFERENCE_TOKEN="secret")
    def test_member_cannot_run_inference(self):
        TeamMembership.objects.filter(user=self.user).update(role=TeamMembershipRoleChoices.TEAM_MEMBER)
        response = self.client.post(
            "/v1/g3/inferences",
            {"stl": SimpleUploadedFile("car.stl", b"solid car\nendsolid car")},
            format="multipart",
            HTTP_X_TEAM_ID=str(self.team.id),
        )
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
