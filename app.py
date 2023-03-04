import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

model_id = "waifu-diffusion/wd-1-5-beta2"
pipe = StableDiffusionPipeline.from_pretrained(model_id, custom_pipeline="lpw_stable_diffusion")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

def app():
    st.title("Waifu Diffusion Streamlit App")
    prompt = st.text_input("Prompt")
    negative_prompt = st.text_input("Negative Prompt")
    num_inference_steps = st.slider("Number of Inference Steps", min_value=1, max_value=50, value=28)
    guidance_scale = st.slider("Guidance Scale", min_value=1, max_value=20, value=10)
    width = st.slider("Image Width", min_value=100, max_value=1000, value=576)
    height = st.slider("Image Height", min_value=100, max_value=1000, value=768)
    max_embeddings_multiples = st.slider("Max Embeddings Multiples", min_value=1, max_value=10, value=2)
    if st.button("Generate Image"):
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            max_embeddings_multiples=max_embeddings_multiples
        ).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    app()
