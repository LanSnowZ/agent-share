# -*- coding: utf-8 -*-
from typing import Optional

from alibabacloud_dm20151123 import models as dm_20151123_models
from alibabacloud_dm20151123.client import Client as Dm20151123Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models

from .config import email_settings


def create_client() -> Optional[Dm20151123Client]:
    """创建阿里云 DirectMail 客户端。"""
    if (
        not email_settings.ALIBABA_CLOUD_ACCESS_KEY_ID
        or not email_settings.ALIBABA_CLOUD_ACCESS_KEY_SECRET
    ):
        print("❌ 未配置 ALIBABA_CLOUD_ACCESS_KEY_ID/ALIBABA_CLOUD_ACCESS_KEY_SECRET")
        return None
    cfg = open_api_models.Config(
        access_key_id=email_settings.ALIBABA_CLOUD_ACCESS_KEY_ID,
        access_key_secret=email_settings.ALIBABA_CLOUD_ACCESS_KEY_SECRET,
    )
    cfg.endpoint = "dm.aliyuncs.com"
    return Dm20151123Client(cfg)


def build_html_body(code: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset=\"UTF-8\" />
      <title>验证码邮件</title>
      <style>
        body {{ font-family: Arial, sans-serif; background:#f5f5f5; margin:0; padding:24px; }}
        .card {{ max-width:600px; margin:0 auto; background:#fff; padding:28px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.06); }}
        .title {{ margin:0 0 12px; font-size:20px; color:#222; }}
        .code {{ margin:18px 0; padding:16px; background:#f8f9fa; border-radius:8px; font-size:30px; letter-spacing:6px; color:#0d6efd; text-align:center; font-weight:bold; }}
        .tips {{ color:#666; font-size:14px; }}
      </style>
    </head>
    <body>
      <div class=\"card\">
        <h1 class=\"title\">{email_settings.SENDER_NAME} 验证码</h1>
        <div class=\"tips\">以下是您的验证码，请在 5 分钟内完成验证：</div>
        <div class=\"code\">{code}</div>
        <div class=\"tips\">如果非本人操作，请忽略此邮件。</div>
      </div>
    </body>
    </html>
    """


def send_email(to_address: str, code: str) -> bool:
    """发送验证码邮件。"""
    client = create_client()
    if client is None:
        return False

    req = dm_20151123_models.SingleSendMailRequest(
        account_name=email_settings.SENDER_EMAIL,
        address_type=1,
        to_address=to_address,
        subject=f"【{email_settings.SENDER_NAME}】{email_settings.EMAIL_SUBJECT}",
        html_body=build_html_body(code),
        reply_to_address=False,
    )

    runtime = util_models.RuntimeOptions()
    try:
        resp = client.single_send_mail_with_options(req, runtime)
        print("✅ 已发送")
        try:
            print(f"📨 RequestId: {resp.body.request_id}")
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"❌ 发送失败: {e}")
        try:
            # 某些异常对象带有 data.Recommend
            recommend = getattr(e, "data", {}).get("Recommend")  # type: ignore[attr-defined]
            if recommend:
                print(f"💡 建议: {recommend}")
        except Exception:
            pass
        return False
