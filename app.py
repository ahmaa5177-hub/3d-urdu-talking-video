import os
import tempfile
import subprocess
from pathlib import Path

import gradio as gr
from huggingface_hub import InferenceClient


# =========================================================
# SETTINGS
# =========================================================

HF_TOKEN = os.getenv("HF_TOKEN")

# Image → Video
I2V_MODEL = "Wan-AI/Wan2.2-I2V-A14B"

# Text → Video
T2V_MODEL = "Wan-AI/Wan2.1-T2V-1.3B"

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN نہیں ملا۔ Hugging Face Space Settings > Secrets میں HF_TOKEN شامل کریں۔"
    )

client = InferenceClient(
    token=HF_TOKEN
)


# =========================================================
# VIDEO DURATION
# =========================================================

DURATIONS = {
    "5 سیکنڈ": 5,
    "10 سیکنڈ": 10,
    "15 سیکنڈ": 15,
    "30 سیکنڈ": 30,
    "60 سیکنڈ": 60,
}


# =========================================================
# SAVE BYTES
# =========================================================

def save_video(video_bytes, extension=".mp4"):
    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    )

    temp.write(video_bytes)
    temp.close()

    return temp.name


# =========================================================
# MAKE VIDEO LONGER
# =========================================================

def set_duration(input_video, duration):
    """
    AI model جو مختصر video بنائے گا،
    اسے مطلوبہ duration تک loop/trim کیا جائے گا۔
    """

    output_video = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name

    command = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        input_video,
        "-t",
        str(duration),
        "-vf",
        "scale=768:-2:force_original_aspect_ratio=decrease,"
        "pad=768:768:(ow-iw)/2:(oh-ih)/2",
        "-r",
        "16",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_video
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg error:\n" +
            result.stderr.decode("utf-8", errors="ignore")
        )

    return output_video


# =========================================================
# IMAGE → VIDEO
# =========================================================

def image_to_video(
    image,
    prompt,
    duration_name,
    progress=gr.Progress()
):

    if image is None:
        raise gr.Error("پہلے تصویر Upload کریں۔")

    if not prompt or not prompt.strip():
        prompt = (
            "A realistic cinematic video, natural movement, "
            "subtle camera motion, highly detailed, realistic lighting"
        )

    duration = DURATIONS[duration_name]

    progress(0.05, desc="تصویر تیار کی جا رہی ہے...")

    try:

        progress(0.15, desc="AI video بن رہی ہے...")

        with open(image, "rb") as f:
            image_bytes = f.read()

        video_bytes = client.image_to_video(
            image=image_bytes,
            model=I2V_MODEL,
            prompt=prompt,
            num_inference_steps=25
        )

        progress(0.75, desc="ویڈیو تیار ہو رہی ہے...")

        original_video = save_video(video_bytes)

        final_video = set_duration(
            original_video,
            duration
        )

        progress(1.0, desc="مکمل!")

        return final_video

    except Exception as e:

        raise gr.Error(
            "Video بنانے میں مسئلہ آیا:\n\n" +
            str(e)
        )


# =========================================================
# TEXT → VIDEO
# =========================================================

def text_to_video(
    prompt,
    duration_name,
    progress=gr.Progress()
):

    if not prompt or not prompt.strip():
        raise gr.Error("Text/Prompt لکھیں۔")

    duration = DURATIONS[duration_name]

    progress(0.05, desc="Prompt تیار ہو رہا ہے...")

    try:

        progress(0.15, desc="AI Text → Video شروع...")

        video_bytes = client.text_to_video(
            prompt=prompt,
            model=T2V_MODEL,
            num_inference_steps=25
        )

        progress(0.75, desc="ویڈیو process ہو رہی ہے...")

        original_video = save_video(video_bytes)

        final_video = set_duration(
            original_video,
            duration
        )

        progress(1.0, desc="مکمل!")

        return final_video

    except Exception as e:

        raise gr.Error(
            "Text سے video بنانے میں مسئلہ آیا:\n\n" +
            str(e)
        )


# =========================================================
# UI
# =========================================================

CSS = """
body {
    direction: rtl;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

.title {
    text-align: center;
    font-size: 34px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    opacity: 0.8;
    margin-bottom: 25px;
}

.generate-btn {
    min-height: 55px !important;
    font-size: 20px !important;
    font-weight: bold !important;
}
"""


with gr.Blocks(
    title="3D Urdu AI Video",
    css=CSS,
    theme=gr.themes.Soft()
) as demo:

    gr.HTML(
        """
        <div class="title">
        🎬 3D Urdu AI Video
        </div>

        <div class="subtitle">
        تصویر یا Text سے AI ویڈیو بنائیں
        </div>
        """
    )

    with gr.Tabs():

        # =================================================
        # IMAGE TAB
        # =================================================

        with gr.Tab("🖼️ تصویر → ویڈیو"):

            with gr.Row():

                with gr.Column():

                    image_input = gr.Image(
                        type="filepath",
                        label="تصویر Upload کریں"
                    )

                    image_prompt = gr.Textbox(
                        label="ویڈیو میں کیا ہونا چاہیے؟",
                        placeholder=(
                            "مثال: آدمی آہستہ چل رہا ہے، "
                            "کیمرہ اس کے ساتھ حرکت کر رہا ہے، "
                            "cinematic realistic video"
                        ),
                        lines=5
                    )

                    image_duration = gr.Dropdown(
                        choices=list(DURATIONS.keys()),
                        value="5 سیکنڈ",
                        label="ویڈیو کا وقت"
                    )

                    image_button = gr.Button(
                        "🎬 تصویر سے ویڈیو بنائیں",
                        variant="primary",
                        elem_classes="generate-btn"
                    )

                with gr.Column():

                    image_output = gr.Video(
                        label="آپ کی AI ویڈیو",
                        autoplay=False
                    )

            image_button.click(
                fn=image_to_video,
                inputs=[
                    image_input,
                    image_prompt,
                    image_duration
                ],
                outputs=image_output
            )

        # =================================================
        # TEXT TAB
        # =================================================

        with gr.Tab("✍️ Text → ویڈیو"):

            with gr.Row():

                with gr.Column():

                    text_prompt = gr.Textbox(
                        label="اپنا Text / Prompt لکھیں",
                        placeholder=(
                            "مثال:\n"
                            "ایک پاکستانی نوجوان کراچی کی سڑک پر "
                            "شام کے وقت چل رہا ہے، cinematic realistic scene"
                        ),
                        lines=8
                    )

                    text_duration = gr.Dropdown(
                        choices=list(DURATIONS.keys()),
                        value="5 سیکنڈ",
                        label="ویڈیو کا وقت"
                    )

                    text_button = gr.Button(
                        "🎬 Text سے ویڈیو بنائیں",
                        variant="primary",
                        elem_classes="generate-btn"
                    )

                with gr.Column():

                    text_output = gr.Video(
                        label="آپ کی AI ویڈیو",
                        autoplay=False
                    )

            text_button.click(
                fn=text_to_video,
                inputs=[
                    text_prompt,
                    text_duration
                ],
                outputs=text_output
            )

    gr.Markdown(
        """
        ### ℹ️ استعمال

        **تصویر → ویڈیو:** تصویر Upload کریں، حرکت کا Prompt لکھیں اور Generate دبائیں۔

        **Text → ویڈیو:** صرف اپنا منظر لکھیں اور Generate دبائیں۔

        ⚠️ AI video generation میں وقت اور GPU/API usage لگ سکتا ہے۔
        """
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    demo.launch()
