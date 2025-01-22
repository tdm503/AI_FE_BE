import gradio as gr
from apis.ocr import ocr_api
from apis.imgclassification import img_classification  

def classify_image_or_ocr(image, mode):
    if mode == "OCR":
        return ocr_api(image)
    else:
        return img_classification(image)

# Tạo giao diện Gradio
demo = gr.Interface(
    fn=classify_image_or_ocr,
    inputs=[
        gr.Image(type="filepath", label="Input Image"),
        gr.Radio(["OCR", "Image Classification"], label="Select Mode")  # Cho phép người dùng chọn chế độ
    ],
    outputs=[
        gr.Text(label="Status"),
        gr.Image(label="Output Image")
    ],
    title="OCR and Image Classification Application",
    description="This is an application for Optical Character Recognition (OCR) and Image Classification.",
)

demo.launch(server_name="127.0.0.1", server_port=3000, share=False)
