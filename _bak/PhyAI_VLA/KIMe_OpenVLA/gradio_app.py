import gradio as gr
import requests
import json_numpy
import numpy as np
from PIL import Image

# Gradio 클라이언트와 서버 간 데이터 포맷 처리
json_numpy.patch()

# REST API 서버 엔드포인트
API_URL = "http://localhost:8000/act"

def predict_action(image, instruction, unnorm_key=None):
    # 업로드된 이미지를 numpy 배열로 변환
    image_array = np.array(image)

    # 요청 데이터(payload) 생성
    payload = {
        "image": image_array,
        "instruction": instruction,
    }

    if unnorm_key:
        payload["unnorm_key"] = unnorm_key

    # 서버에 POST 요청
    response = requests.post(API_URL, json=payload)
    
    # 서버 응답 확인
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error {response.status_code}: {response.text}"

# Gradio 인터페이스 구성
with gr.Blocks() as gradio_app:
    gr.Markdown("# OpenVLA Robot Action Prediction")
    gr.Markdown(
        "Provide an image of the robot workspace and an instruction to predict the robot's action. "
        "You can either upload an image or provide a base64-encoded image via API."
    )

    with gr.Row():
        with gr.Column(scale=1):
            instruction_input = gr.Textbox(label="Instruction", placeholder="e.g., Pick up the remote")
            unnorm_key_input = gr.Textbox(label="Unnorm Key (Optional)", placeholder="e.g., bridge_orig")
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Submit")

        with gr.Column(scale=1):
            output_action = gr.Textbox(label="Robot Action (X, Y, Z, Roll, Pitch, Yaw, Gripper)", interactive=False, lines=8)
    

    # 예측 함수 연결
    submit_btn.click(
        fn=predict_action,
        inputs=[image_input, instruction_input, unnorm_key_input],
        outputs=[output_action]
    )

    # 예제 제공
    gr.Examples(
        examples=[
            ["Place the red vegetable in the silver pot.", "bridge_orig", "./KIMe_OpenVLA/images/example1.jpeg"],
            ["Pick up the remote", "bridge_orig", "./KIMe_OpenVLA/images/example2.jpeg"]
        ],
        inputs=[instruction_input, unnorm_key_input, image_input]
    )

gradio_app.launch()