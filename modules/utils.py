# modules/utils.py
import timm

def load_model(model_name, pretrained=True):
    """
    timm 라이브러리를 이용해 모델 생성.
    - 먼저 pretrained=True로 시도하고, RuntimeError가 발생하면 fallback으로 pretrained=False 재시도.
    - 최종 pretrained 여부를 print 로그에 기록.
    """
    try:
        model = timm.create_model(model_name, pretrained=pretrained)
        print(f"[Info] Loaded model='{model_name}' with pretrained={pretrained}")
    except RuntimeError as e:
        print(f"[Warning] Failed to load '{model_name}' with pretrained={pretrained}. Error: {e}")
        print("         Fallback to pretrained=False.")
        model = timm.create_model(model_name, pretrained=False)
        print(f"[Info] Reloaded model='{model_name}' with pretrained=False")

    return model
