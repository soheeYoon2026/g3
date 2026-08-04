import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from gunicorn.http import wsgi
except (ImportError, OSError):
    wsgi = None

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

load_dotenv(BASE_DIR / ".env")

PROJECT_NAME = "adro"
SITE_NAME = "adro"
SITE_LOGO = "img/logo.png"  # app/staticfiles/static/img/logo.png 변경
DOMAIN = os.environ.get("DOMAIN", "aoxadro.com")
AWS_S3_CHECKSUM_ALGORITHM = os.environ.get("AWS_S3_CHECKSUM_ALGORITHM", "CRC64NVME")
G3_INFERENCE_URL = os.environ.get("G3_INFERENCE_URL", "")
G3_INFERENCE_TOKEN = os.environ.get("G3_INFERENCE_TOKEN", "")
G3_INFERENCE_TIMEOUT_SECONDS = int(os.environ.get("G3_INFERENCE_TIMEOUT_SECONDS", "300"))
G3_INFERENCE_CONFIG_BUCKET = os.environ.get("G3_INFERENCE_CONFIG_BUCKET", "")
G3_INFERENCE_CONFIG_KEY = os.environ.get("G3_INFERENCE_CONFIG_KEY", "_private/g3/inference-service.json")

# Large CAD/mesh request bodies: STEP files arrive as multipart through the BFF
# (/v1/stl/convert) and the repair fallback sends base64 STL JSON. Django's
# 2.5MB default raises RequestDataTooBig on anything bigger (a 2.8MB .stp
# failed). Aligned with the endpoints' 200MB per-file cap, plus headroom for
# base64/multipart overhead. Files past FILE_UPLOAD_MAX_MEMORY_SIZE stream to a
# temp file as usual.
DATA_UPLOAD_MAX_MEMORY_SIZE = 256 * 1024 * 1024


if wsgi is not None:

    class Response(wsgi.Response):
        def __init__(self, req, sock, cfg):
            super(Response, self).__init__(req, sock, cfg)
            self.version = PROJECT_NAME

    wsgi.Response = Response


LOCAL_APPS = [
    "app.staticfile",
    "app.common.apps.CommonConfig",
    "app.admin_user.apps.AdminUserConfig",
    "app.artifacts.apps.ArtifactsConfig",
    "app.book_demo.apps.BookDemoConfig",
    "app.celery_log.apps.CeleryLogConfig",
    # "app.control_plane.apps.ControlPlaneConfig",
    "app.device.apps.DeviceConfig",
    "app.email_log.apps.EmailLogConfig",
    "app.jobs.apps.JobsConfig",
    # "app.ops_plane.apps.OpsPlaneConfig",  # todo
    "app.patch.apps.PatchConfig",
    "app.projects.apps.ProjectsConfig",
    "app.sms_log.apps.SmsLogConfig",
    "app.user.apps.UserConfig",
    "app.verifier.apps.VerifierConfig",
    "app.withdrawal_user.apps.WithdrawalUserConfig",
    "app.presigned_url.apps.PreSignedUrlConfig",
    "app.tenants.apps.TenantsConfig",
    "app.payments.apps.PaymentsConfig",
    "app.credits.apps.CreditsConfig",
    "app.plans.apps.PlansConfig",
]

THIRD_PARTY_APPS = [
    "rest_framework",
    "corsheaders",
    "django_filters",
    "django_hosts",
    "drf_spectacular",
    "drf_spectacular_sidecar",
    "storages",
    "django_celery_results",
    "ordered_model",
]

DJANGO_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
]

INSTALLED_APPS = LOCAL_APPS + THIRD_PARTY_APPS + DJANGO_APPS

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "config.middleware.SwaggerLoginMiddleware",
    "django_hosts.middleware.HostsRequestMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "app.middleware.tenant.TenantMiddleware",  # Tenant middleware after SecurityMiddleware
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "app.common.csrf.ApiCsrfMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "config.middleware.RequestLogMiddleware",
]


TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# Password validation
# https://docs.djangoproject.com/en/3.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/3.1/topics/i18n/

LANGUAGE_CODE = "ko"

TIME_ZONE = "Asia/Seoul"

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.1/howto/static-files/


# APPEND_SLASH
APPEND_SLASH = False

# AUTH_USER_MODEL
AUTH_USER_MODEL = "admin_user.AdminUser"

# APPLICATION
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"


# HOST
DEFAULT_HOST = "api"
ROOT_HOSTCONF = "config.hosts"
ROOT_URLCONF = "config.urls.api"


# MODEL ID
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# DATABASE ROUTER
DATABASE_ROUTERS = ["app.middleware.db_router.TenantDBRouter", "config.router.Router"]


# SIMPLE_JWT
SIMPLE_JWT_ALGORITHM = "HS256"


# AUTH — 공용 로그인(auth.aoxadro.com) → 테넌트 콜백 리다이렉트 설정
OTT_TTL_SECONDS = int(os.environ.get("OTT_TTL_SECONDS", "60"))
AUTH_REDIRECT_SCHEME = os.environ.get("AUTH_REDIRECT_SCHEME", "https")
AUTH_REDIRECT_HOST_TEMPLATE = os.environ.get(
    "AUTH_REDIRECT_HOST_TEMPLATE",
    "{slug}.aoxadro.com",
)
AUTH_CALLBACK_PATH = os.environ.get("AUTH_CALLBACK_PATH", "/callback")
# AUTH_SIGNUP_PATH = os.environ.get("AUTH_SIGNUP_PATH", "/signup")
# AUTH_RESET_PASSWORD_PATH = os.environ.get("AUTH_RESET_PASSWORD_PATH", "/reset-password")


# djangorestframework-camel-case: prevent underscore insertion before digits
# Without this, UUID dict keys get mangled (e.g. "29f8b81265bc" → "29f8b_81265bc")
#
# ignore_fields: 해당 키의 "값(중첩 dict)" 은 재귀 변환에서 제외.
#   - runConfig/config 는 FE 가 보낸 키를 HPC 그대로 전달해야 함.
#     예: G1 의 "R", "U" 같은 단일 대문자 키가 underscoreize 의 .lower() 로
#         "r", "u" 가 되어 sanitise_g1_config 의 default("R": 0.15) 와 충돌,
#         같은 dict 안에 R 과 r 이 동시에 존재하는 회귀가 있었음.
JSON_CAMEL_CASE = {
    "JSON_UNDERSCOREIZE": {
        "no_underscore_before_number": True,
        "ignore_fields": ("config", "runConfig", "run_config", "spec"),
    }
}

# DJANGO REST FRAMEWORK
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "app.common.authentication.Authentication",
    ],
    "DEFAULT_RENDERER_CLASSES": [
        "djangorestframework_camel_case.render.CamelCaseJSONRenderer",
        "djangorestframework_camel_case.render.CamelCaseBrowsableAPIRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "djangorestframework_camel_case.parser.CamelCaseJSONParser",
        "djangorestframework_camel_case.parser.CamelCaseFormParser",
        "djangorestframework_camel_case.parser.CamelCaseMultiPartParser",
    ],
    "DEFAULT_PAGINATION_CLASS": "app.common.pagination.LimitOffsetPagination",
    "DEFAULT_FILTER_BACKENDS": [
        "app.common.filter_backends.FilterBackend",
        "app.common.filter_backends.OrderingFilter",
    ],
    "EXCEPTION_HANDLER": "app.common.views.exception_handler",
    "DEFAULT_SCHEMA_CLASS": "app.common.openapi.CustomAutoSchema",
    "TEST_REQUEST_DEFAULT_FORMAT": "json",
    "NON_FIELD_ERRORS_KEY": "non_field",
    "ENABLE_LIST_MECHANICS_ON_NON_2XX": [400],
}


# SPECTACULAR
SPECTACULAR_SETTINGS = {
    "SWAGGER_UI_DIST": "SIDECAR",
    "SWAGGER_UI_FAVICON_HREF": "SIDECAR",
    "REDOC_DIST": "SIDECAR",
    "TITLE": f"{SITE_NAME} API",
    "DESCRIPTION": f"""개발: [api.dev.{DOMAIN}](https://api.dev.{DOMAIN})<br/>운영: [api.{DOMAIN}](https://api.{DOMAIN})""",
    "VERSION": "1.0.0",
    "SCHEMA_PATH_PREFIX": "/v[0-9]",
    "DISABLE_ERRORS_AND_WARNINGS": True,
    "SORT_OPERATIONS": False,
    "SWAGGER_UI_SETTINGS": {
        "docExpansion": "none",
        "defaultModelRendering": "model",
        "defaultModelsExpandDepth": 0,
        "deepLinking": True,
        "displayRequestDuration": True,
        "persistAuthorization": True,
        "syntaxHighlight.activate": True,
        "syntaxHighlight.theme": "agate",
        "showExtensions": True,
        "filter": True,
    },
    "SERVE_INCLUDE_SCHEMA": False,
    "APPEND_COMPONENTS": {
        "securitySchemes": {
            "Bearer": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "Enter: Bearer <your-token>",
            },
        },
    },
    "SECURITY": [
        {
            "Bearer": [],
        },
    ],
    "PREPROCESSING_HOOKS": [
        "app.common.spectacular_hooks.api_ordering",
    ],
    "POSTPROCESSING_HOOKS": [
        "drf_spectacular.contrib.djangorestframework_camel_case.camelize_serializer_fields",
        "app.common.spectacular_hooks.mark_optional_response_fields",
    ],
    "COMPONENT_SPLIT_REQUEST": True,
    "ENUM_ADD_EXPLICIT_BLANK_NULL_CHOICE": False,
    "SORT_OPERATION_PARAMETERS": False,
}


# AWS SES
DEFAULT_FROM_EMAIL = f"noreply@{DOMAIN}"


# CKEDITOR
SECURE_REFERRER_POLICY = "strict-origin-when-cross-origin"
CKEDITOR_5_CONFIGS = {
    "default": {
        "toolbar": {
            "items": [
                "heading",
                "|",
                "bold",
                "link",
                "imageUpload",
            ],
        }
    },
}


# ── G1 Multi-node Configuration ──────────────────────────────────────────────
# G1 HPC 노드별 SSH 접속 정보 + S3 버킷 매핑은 AWS Secrets Manager 에서 로드.
#   Secret: adro/{env}/g1-nodes  (env var G1_NODES_SECRET_ID 로 override 가능)
#   노드 키: node1, node2, node3 (각 노드의 `region` 필드가 AWS 리전 지정)
#
# 런타임 조회는 app.jobs.g1_node_selector.get_g1_nodes_config() 사용.
# 로컬 개발 편의상 SM 조회 실패 시 .env 의 단일 G1_HPC_SSH_* 를 node1 로 fallback.

G1_NODES_SECRET_ID = os.environ.get("G1_NODES_SECRET_ID", "")


# CELERY
CELERY_ENABLE_UTC = False
CELERY_TIMEZONE = "Asia/Seoul"
CELERYD_SOFT_TIME_LIMIT = 3600  # 1시간 (G1 STL 1.4GB+ 업로드 시간 고려)
CELERYD_TIME_LIMIT = CELERYD_SOFT_TIME_LIMIT + 60
CELERY_TASK_ACKS_LATE = True
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
CELERY_WORKER_MAX_TASKS_PER_CHILD = 500

# Shared cache. Batch-submit dedupe uses django.core.cache.add(), so deployed
# multi-worker processes must share the same backend. Local/test environments
# keep LocMemCache unless a Redis URL is explicitly injected.
_DJANGO_CACHE_URL = (
    os.environ.get("DJANGO_CACHE_URL")
    or os.environ.get("CACHE_URL")
    or (os.environ.get("CELERY_BROKER_URL") if os.environ.get("CELERY_BROKER_URL") else "")
)
if _DJANGO_CACHE_URL:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": _DJANGO_CACHE_URL,
            "KEY_PREFIX": os.environ.get(
                "DJANGO_CACHE_KEY_PREFIX",
                f"{PROJECT_NAME}:{os.environ.get('CONFIG_ENV') or os.environ.get('APP_ENV') or 'local'}",
            ),
            "TIMEOUT": int(os.environ.get("DJANGO_CACHE_TIMEOUT", "300")),
        }
    }
else:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "aox-django-local-cache",
        }
    }

# Usage event ingestion (G1/G2/G4 -> Lambda -> SQS FIFO)
USAGE_EVENT_ENABLED = os.environ.get("USAGE_EVENT_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
USAGE_EVENT_ENDPOINT = os.environ.get("USAGE_EVENT_ENDPOINT", "")
USAGE_EVENT_TOKEN_SECRET = os.environ.get("USAGE_EVENT_TOKEN_SECRET", "")
USAGE_EVENT_TOKEN_TTL_SECONDS = int(os.environ.get("USAGE_EVENT_TOKEN_TTL_SECONDS", "604800"))
USAGE_EVENT_TIMEOUT_SECONDS = float(os.environ.get("USAGE_EVENT_TIMEOUT_SECONDS", "3.0"))
USAGE_EVENT_PRICING_VERSION = os.environ.get("USAGE_EVENT_PRICING_VERSION", "pricing_v0.0")
USAGE_EVENT_TICK_INTERVAL_SECONDS = os.environ.get("USAGE_EVENT_TICK_INTERVAL_SECONDS", "")
CREDIT_USAGE_EVENTS_QUEUE_URL = os.environ.get("CREDIT_USAGE_EVENTS_QUEUE_URL", "")
CREDIT_USAGE_EVENTS_QUEUE_ARN = os.environ.get("CREDIT_USAGE_EVENTS_QUEUE_ARN", "")
CREDIT_USAGE_EVENTS_DLQ_URL = os.environ.get("CREDIT_USAGE_EVENTS_DLQ_URL", "")
CREDIT_USAGE_EVENTS_DLQ_ARN = os.environ.get("CREDIT_USAGE_EVENTS_DLQ_ARN", "")

# Realtime usage-event consumer toggle. Off by default so test/local
# environments don't poll SQS. dev/prod ECS sets this to "true" via Terraform.
USAGE_EVENT_CONSUMER_ENABLED = os.environ.get("USAGE_EVENT_CONSUMER_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# Long-running task self-termination (sec). Pairs with beat schedule 540s
# below to maintain a 60s overlap between successive workers.
USAGE_EVENT_CONSUMER_RUN_SECONDS = int(os.environ.get("USAGE_EVENT_CONSUMER_RUN_SECONDS", "600"))
# Visibility timeout overrides the queue default (60s in Terraform) — our
# handle_event can sit on a row lock long enough that 60s isn't safe.
USAGE_EVENT_VISIBILITY_TIMEOUT = int(os.environ.get("USAGE_EVENT_VISIBILITY_TIMEOUT", "120"))
USAGE_EVENT_WAIT_TIME_SECONDS = int(os.environ.get("USAGE_EVENT_WAIT_TIME_SECONDS", "20"))
USAGE_EVENT_MAX_MESSAGES = int(os.environ.get("USAGE_EVENT_MAX_MESSAGES", "10"))
USAGE_EVENT_SQS_ERROR_SLEEP = float(os.environ.get("USAGE_EVENT_SQS_ERROR_SLEEP", "5.0"))

# CELERY BEAT SCHEDULE
CELERY_BEAT_SCHEDULE = {
    "poll-g2-job-status": {
        "task": "app.tasks.g2.poll_g2_job_status",
        "schedule": 30.0,
    },
    # 삭제(soft-delete) 후 유예기간 지난 잡의 S3 산출물 정리 (잡당 ~1GB 방지)
    "purge-deleted-job-s3": {
        "task": "app.tasks.maintenance.purge_deleted_job_s3",
        "schedule": 86400.0,
    },
    "poll-g4-job-status": {
        "task": "app.tasks.g4.poll_g4_job_status",
        "schedule": 30.0,
    },
    "poll-g1-job-status": {
        "task": "app.tasks.g1.poll_g1_job_status",
        "schedule": 30.0,
    },
    "dispatch-queued-g1-jobs": {
        "task": "app.tasks.g1.dispatch_queued_g1_jobs",
        "schedule": 30.0,
    },
    "check-g1-health": {
        "task": "app.tasks.health.check_g1_health",
        "schedule": 30.0,
    },
    "check-g2-health": {
        "task": "app.tasks.health.check_g2_health",
        "schedule": 30.0,
    },
    "cleanup-old-hpc-jobs": {
        "task": "app.tasks.g1.cleanup_old_hpc_jobs",
        "schedule": 86400.0,  # 24시간 (1일 1회)
    },
    "cleanup-orphaned-s3-files": {
        "task": "app.artifacts.tasks.cleanup_orphaned_s3_files",
        "schedule": 86400.0,  # 24시간 (1일 1회)
    },
    "purge-soft-deleted-jobs": {
        "task": "app.tasks.jobs.purge_soft_deleted_jobs",
        "schedule": 86400.0,  # 24시간 (1일 1회)
    },
    "deactivate-expired-tenants": {
        "task": "app.tenants.tasks.deactivate_expired_tenants",
        "schedule": 60.0,  # 1분 (사용 종료일 도달 시 빠르게 비활성화)
    },
    "expire-credit-buckets": {
        "task": "app.credits.tasks.expire_due_buckets",
        "schedule": 86400.0,  # 24시간 (1일 1회 자정 부근). 조회 시 lazy filter 가 보조.
    },
    "run-subscription-billing": {
        "task": "app.plans.tasks.run_subscription_billing",
        "schedule": 300.0,  # 5분 (next_billing_attempt_at 도래 + PAST_DUE 5h 재시도 후보 탐색).
    },
    "finalize-subscription-cycles": {
        "task": "app.plans.tasks.finalize_subscription_cycles",
        "schedule": 300.0,  # 5분 (사이클 만료/SCHEDULED 활성 전환).
    },
    "reconcile-stale-payments": {
        "task": "app.payments.tasks.reconcile_stale_payments",
        "schedule": 900.0,  # 15분 (READY/PENDING 5분 이상 결제건 PortOne 재조회 + drift 복구).
    },
    "reconcile-unfulfilled-post-payments": {
        "task": "app.payments.tasks.reconcile_unfulfilled_post_payments",
        "schedule": 600.0,  # 10분 (PAID 됐지만 사이클 활성/credit 발급 누락된 결제 재발화).
    },
    "consume-usage-events": {
        "task": "app.credits.tasks.consume_usage_events",
        # 540s schedule + 600s long-run = 항상 60s 겹침 → 갭 0 +
        # 한 워커 crash 시 다른 워커가 즉시 takeover.
        "schedule": 540.0,
        "options": {"queue": "usage_consumer"},
    },
}

# Route lightweight submit/poll wrappers explicitly. Heavy mesh processing runs
# in AWS Batch via manifest bridge, not local Celery workers.
CELERY_TASK_ROUTES = {
    "app.patch.tasks.poll_source_mesh_processor_batch_manifest": {"queue": "celery"},
    "app.tasks.jobs.build_job_result_preview_artifacts": {"queue": "celery"},
    "app.tasks.jobs.poll_result_preview_batch_manifest": {"queue": "celery"},
    "app.credits.tasks.consume_usage_events": {"queue": "usage_consumer"},
}

# SOCIAL
SOCIAL_REDIRECT_PATH = "/social/callback"

# ── PortOne (V2) ─────────────────────────────────────────────────────────────
# 결제 모듈 설정. base.py 는 환경변수에서 읽고, 미설정이면 빈 문자열 → Django
# boot 는 통과한다. 실제 결제/빌링 호출 시점에 값이 비어있으면 서비스 레이어가
# 명시적 에러를 raise.
#
# 운영(dev/prod) 환경에서는 AWS Secrets Manager 로 덮어쓰는 것을 권장:
#   - 시크릿 ID 형식: "{PROJECT_NAME}/{APP_ENV}/portone"  (예: adro/dev/portone)
#   - 시크릿 JSON 스키마 (멀티-PG):
#       {
#         "store_id":              "store-xxxx",
#         "channel_key_payletter":       "channel-key-xxxx-payletter",
#         "channel_key_paypal":          "channel-key-xxxx-paypal",
#         "channel_key_galaxia":         "channel-key-xxxx-galaxia (신용카드_인증_… MID, 결제창 결제)",
#         "channel_key_galaxia_billing": "channel-key-xxxx-galaxia-billing (신용카드_빌링 MID)",
#         "api_secret":                  "...",
#         "webhook_secret":              "whsec_..."
#       }
#   - 실제 override 는 config/settings/dev.py, prod.py 의 PORTONE 블록 참조.
#
# 로컬 개발(local/local_docker)에서는 .env 의 환경변수만으로 충분.
#
# 필요한 환경변수 (로컬 또는 폴백):
#   PORTONE_STORE_ID              - 포트원 상점 ID (콘솔 > 상점 설정)
#   PORTONE_CHANNEL_KEY_PAYLETTER - PayLetter 채널 키 (국내·해외 카드)
#   PORTONE_CHANNEL_KEY_PAYPAL    - PayPal V2 (SPB/RT) 채널 키
#   PORTONE_CHANNEL_KEY_GALAXIA         - Galaxia 단건결제 채널 키 (신용카드_수기 MID)
#   PORTONE_CHANNEL_KEY_GALAXIA_BILLING - Galaxia 빌링키 발급/정기결제 채널 키 (신용카드_빌링 MID)
#   PORTONE_API_SECRET            - 서버용 API Secret (콘솔 > API 키)
#   PORTONE_WEBHOOK_SECRET        - 웹훅 시크릿 (콘솔 > 결제알림(Webhook) 관리)
#   PORTONE_API_BASE_URL          - 기본값 https://api.portone.io
#
PORTONE_STORE_ID = os.environ.get("PORTONE_STORE_ID", "")

PORTONE_CHANNEL_KEY_PAYLETTER = os.environ.get("PORTONE_CHANNEL_KEY_PAYLETTER", "")
PORTONE_CHANNEL_KEY_PAYPAL = os.environ.get("PORTONE_CHANNEL_KEY_PAYPAL", "")
# Galaxia 는 PG 상점아이디(MID) 가 결제 유형별로 분리 발급된다:
#   - 신용카드_인증_휴대폰_계좌이체_가상계좌 MID → 결제창 인증결제용 채널 (requestPayment).
#       현재는 payMethod=CARD 만 사용하지만 같은 MID 로 TRANSFER/VIRTUAL_ACCOUNT/MOBILE 확장 가능.
#   - 신용카드_빌링 MID → 카드 빌링키 발급/정기결제용 채널 (requestIssueBillingKey + charge).
# PortOne 콘솔에서 채널 = MID 1:1 이므로 채널키도 2개로 분리해서 보관한다.
# 주의: "신용카드_수기" MID 는 비인증(키인) 결제용 — requestPayment(결제창 인증결제)와 다르므로
#       결제 채널에 쓰면 안 된다. 결제 채널은 반드시 "신용카드_인증_..." MID 로 발급.
# (PayPal/PayLetter 는 MID 를 안 나눠 단일 채널로 결제+빌링 모두 처리)
PORTONE_CHANNEL_KEY_GALAXIA = os.environ.get("PORTONE_CHANNEL_KEY_GALAXIA", "")
PORTONE_CHANNEL_KEY_GALAXIA_BILLING = os.environ.get("PORTONE_CHANNEL_KEY_GALAXIA_BILLING", "")

# Provider → 결제(단건)용 channel key 매핑. settings.PORTONE_CHANNEL_KEYS[provider] 로 lookup.
PORTONE_CHANNEL_KEYS = {
    "payletter": PORTONE_CHANNEL_KEY_PAYLETTER,
    "paypal": PORTONE_CHANNEL_KEY_PAYPAL,
    "galaxia": PORTONE_CHANNEL_KEY_GALAXIA,
}

# Provider → 빌링키 발급/정기결제용 channel key 매핑. 빌링 MID 가 결제 MID 와 다른 PG
# (Galaxia) 만 별도 채널을 갖고, 단일 채널 PG (PayPal/PayLetter) 는 결제 채널로 fallback.
# 빌링키 발급 SDK 호출 (requestIssueBillingKey) 과 bypass-hint endpoint 가 이 매핑을 사용.
PORTONE_BILLING_CHANNEL_KEYS = {
    "payletter": PORTONE_CHANNEL_KEY_PAYLETTER,
    "paypal": PORTONE_CHANNEL_KEY_PAYPAL,
    "galaxia": PORTONE_CHANNEL_KEY_GALAXIA_BILLING or PORTONE_CHANNEL_KEY_GALAXIA,
}

# 멀티-통화 기본값. PayLetter 는 KRW/USD/JPY, PayPal 은 USD 만, Galaxia 는 KRW 만.
# request serializer 화이트리스트와 동기화해서 사용.
PORTONE_DEFAULT_CURRENCY = os.environ.get("PORTONE_DEFAULT_CURRENCY", "USD")

PORTONE_API_SECRET = os.environ.get("PORTONE_API_SECRET", "")
PORTONE_WEBHOOK_SECRET = os.environ.get("PORTONE_WEBHOOK_SECRET", "")
PORTONE_API_BASE_URL = os.environ.get("PORTONE_API_BASE_URL", "https://api.portone.io")
PORTONE_BROWSER_SDK_URL = os.environ.get(
    "PORTONE_BROWSER_SDK_URL",
    "https://cdn.portone.io/v2/browser-sdk.js",
)

# 결제 테스트 페이지(/payments/test/) 접근 허용 여부.
# 운영 환경에서는 노출 금지(default False). 환경별 settings 에서 명시적으로 True.
PAYMENT_TEST_PAGE_ENABLED = False


# GOOGLE
GOOGLE_CLIENT_ID = "**"
GOOGLE_CLIENT_SECRET = "**"
GOOGLE_LOGIN_URL = "https://accounts.google.com/o/oauth2/auth/oauthchooseaccount?response_type=code&client_id={google_client_id}&redirect_uri={SOCIAL_REDIRECT_URL}&state=google&scope=openid"
