from django.urls import include, path

from app.artifacts.v1 import views as artifact_views
from app.g3_inference.views import create_inference
from app.tenants.v1.views import update_team_billing_email

urlpatterns = [
    path("v1/g3/inferences", create_inference, name="g3-create-inference"),
    path("v1/", include("app.admin_user.v1.urls")),
    path("v1/projects/<int:project_id>/source-uploads/init", artifact_views.init_upload, name="init_upload"),
    path(
        "v1/projects/<int:project_id>/source-uploads/complete",
        artifact_views.complete_upload,
        name="complete_upload",
    ),
    path("v1/projects/<int:project_id>/source-uploads/abort", artifact_views.abort_upload, name="abort_upload"),
    path(
        "v1/projects/<int:project_id>/thumbnail-uploads/init",
        artifact_views.init_thumbnail_upload,
        name="init_thumbnail_upload",
    ),
    path(
        "v1/projects/<int:project_id>/thumbnail-uploads/complete",
        artifact_views.complete_thumbnail_upload,
        name="complete_thumbnail_upload",
    ),
    path(
        "v1/projects/<int:project_id>/thumbnail-uploads/abort",
        artifact_views.abort_thumbnail_upload,
        name="abort_thumbnail_upload",
    ),
    path("v1/", include("app.book_demo.v1.urls")),
    # path("v1/", include("app.control_plane.v1.urls")),  # Removed - no subscription billing implementation yet
    path("v1/", include("app.jobs.v1.urls")),
    path("v1/", include("app.patch.v1.urls")),
    # path("v1/", include("app.ops_plane.v1.urls")),  # Removed - models commented out
    path("v1/", include("app.projects.v1.urls")),
    path("v1/", include("app.user.v1.urls")),
    path("v1/", include("app.verifier.v1.urls")),
    path("v1/", include("app.presigned_url.v1.urls")),
    path("v1/", include("app.payments.v1.urls")),
    path("v1/", include("app.credits.v1.urls")),
    path("v1/", include("app.plans.v1.urls")),
    # 결제 모듈 비-versioned 진입점: 테스트 페이지 + PortOne 웹훅
    path("", include("app.payments.urls")),
    # 크레딧 비-versioned 진입점: SQS 실시간 차감 검증용 테스트 페이지
    path("", include("app.credits.urls")),
    # 팀 결제 이메일(billing_email) 수정용. billing_email 은 빌링키 발급 시 요청자
    # 이메일로 자동 seed 되며, 본 엔드포인트로 대표 이메일을 변경할 수 있다.
    # (TenantsViewSet 라우터는 미사용이라 billing-email 경로만 등록.)
    path(
        "v1/teams/<int:team_id>/billing-email",
        update_team_billing_email,
        name="teams-billing-email-update",
    ),
    # path("v1/", include("app.tenants.v1.urls")),  # todo - 나중에 필요하면 살리기 (0423 기준 사용 X)
]
