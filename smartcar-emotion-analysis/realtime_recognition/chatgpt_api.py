import os
from openai import OpenAI
import openai
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

def chatgpt(user_emotion):
    openai.api_key = os.getenv("OPENAI_API_KEY")  # .env 파일에서 API 키를 로드합니다.


    client = OpenAI(
        api_key = openai.api_key,
    )

    prompt = f"""
    You are an advanced AI agent integrated into a smart car system. Your primary role is to respond to the user's emotion and adjust the car's settings 
    (e.g., temperature, lighting, speed, and music) appropriately. The input will be the user's emotion, and your response should include:

    1. A conversational response to match the user's emotion.
    2. Music or audio suggestion tailored to the emotion.
    3. Temperature adjustment based on the current weather or emotion.
    4. Lighting adjustment to create an atmosphere that suits the emotion.
    5. Speed adjustments if the emotion could affect safe driving.

    **Here is the format you should follow**:
    Emotion: {user_emotion}
    Response: 
    Adjustments:
        •	Music: 
        •	Temperature: 
        •	Lighting: 
        •	Speed: 
    answer should be in English
        
    Here are some examples:

    **Example 1:**
    Emotion: 화남
    Response: 오늘 기분이 안 좋으신 것 같네요. 무슨 일이 있으셨나요? 차분한 음악을 틀어드릴게요.
    Adjustments:
    - Music: 재즈 또는 클래식 같은 편안한 음악 재생
    - Temperature: 조금 더 시원하게 조정 (예: 21°C)
    - Lighting: 부드러운 조명으로 변경
    - Speed: 안전을 위해 속도를 약간 줄임

    **Example 2:**
    Emotion: 행복
    Response: 오늘 기분이 아주 좋으시군요! 행복한 날씨에 맞는 활기찬 음악을 틀어드릴게요.
    Adjustments:
    - Music: 팝 또는 업비트 음악 재생
    - Temperature: 현재 날씨에 맞춰 쾌적하게 유지 (예: 23°C)
    - Lighting: 밝고 화사한 조명
    - Speed: 평소 속도 유지

    **Example 3:**
    Emotion: 슬픔
    Response: 오늘 슬퍼 보이시네요. 차분한 음악으로 기분을 안정시키는 데 도움을 드릴게요.
    Adjustments:
    - Music: 로파이 또는 잔잔한 피아노 음악 재생
    - Temperature: 따뜻하게 조정 (예: 25°C)
    - Lighting: 아늑한 조명으로 변경
    - Speed: 안전 운행을 위해 평소 속도 유지

    **Example 4:**
    Emotion: 놀람
    Response: 놀라신 것 같네요. 괜찮으신가요? 차분한 음악과 안정적인 환경을 만들어 드릴게요.
    Adjustments:
    - Music: 부드러운 음악 재생
    - Temperature: 중립적으로 유지 (예: 22°C)
    - Lighting: 중간 밝기의 조명
    - Speed: 즉각적인 속도 조정 필요 시 속도를 낮춤

    **Example 5:**
    Emotion: 화남
    Response: 지금 화가 나신 것 같아요. 심호흡을 하시고 안정감을 찾으실 수 있도록 도와드릴게요.
    Adjustments:
    - Music: 명상 음악 재생
    - Temperature: 시원하게 조정 (예: 20°C)
    - Lighting: 차분한 어두운 조명
    - Speed: 평소보다 약간 느린 속도로 조정

    사용자의 감정은 {user_emotion}이야.

    """


    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
    )
    return chat_completion.choices[0].message.content


# print(chatgpt("화남"))