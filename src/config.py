# -*- coding: utf-8 -*-
from pydantic_settings import BaseSettings


class CachePathSettings(BaseSettings):
    """缓存配置类"""

    USER_FILE_PATH: str = "user.json"
    MEMORYOS_DATA_DIR: str = "eval/memoryos_data"
    MEMORY_FILE_PATH: str = "../sharememory_user/data/memory.json"

    class Config:
        env_file = ".env"
        extra = "ignore"


class EmailSettings(BaseSettings):
    """邮件配置类"""

    ALIBABA_CLOUD_ACCESS_KEY_ID: str = ""
    ALIBABA_CLOUD_ACCESS_KEY_SECRET: str = ""
    SENDER_EMAIL: str = "noreply@baijia.online"
    SENDER_NAME: str = "Your App Name"
    EMAIL_SUBJECT: str = "验证码邮件"

    class Config:
        env_file = ".env"
        extra = "ignore"


cache_path_settings = CachePathSettings()
email_settings = EmailSettings()
