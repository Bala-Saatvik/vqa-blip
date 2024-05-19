import streamlit as st
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
import numpy as np
import cv2
import speech_recognition as sr

# Initialize BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

class BLIP_VQA:
    def __init__(self, vision_model, text_encoder, text_decoder, processor):
        self.vision_model = vision_model
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        self.processor = processor
    
    def preprocess(self, img, ques):
        inputs = self.processor(img, ques, return_tensors='pt')
        pixel_values = inputs['pixel_values']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        return pixel_values, input_ids, attention_mask
    
    def generate_output(self, pixel_values, input_ids, attention_mask, max_new_tokens=50):
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        img_embeds = vision_outputs[0]
        img_attention_mask = torch.ones(img_embeds.size()[: -1], dtype=torch.long)
        question_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=img_embeds,
            encoder_attention_mask=img_attention_mask,
            return_dict=False
        )
        question_embeds = question_outputs[0]
        question_attention_mask = torch.ones(question_embeds.size()[:-1], dtype=torch.long)
        bos_ids = torch.full((question_embeds.size(0), 1), fill_value=30522)
        outputs = self.text_decoder.generate(
            input_ids=bos_ids,
            eos_token_id=102,
            pad_token_id=0,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=question_attention_mask,
            max_new_tokens=max_new_tokens
        )
        return outputs
    
    def postprocess(self, outputs):
        return self.processor.decode(outputs[0], skip_special_tokens=True)
    
    def get_answer(self, image, ques):
        pixel_values, input_ids, attention_mask = self.preprocess(image, ques)
        outputs = self.generate_output(pixel_values, input_ids, attention_mask)
        answer = self.postprocess(outputs)
        return answer

blip_vqa = BLIP_VQA(
    vision_model=model.vision_model,
    text_encoder=model.text_encoder,
    text_decoder=model.text_decoder,
    processor=processor
)

def get_voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening for Prompt...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio)
        return command
    except sr.UnknownValueError:
        st.write("Could not understand audio.")
        return None
    except sr.RequestError as e:
        st.write("Could not request results; {0}".format(e))
        return None

def segment_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on the original image
    segmented_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
    return segmented_image

# Streamlit UI
st.title("Eye Cloud")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Perform segmentation using OpenCV
    image_np = np.array(image)
    segmented_image = segment_image(image_np)
    st.image(segmented_image, caption='Segmented Image.', use_column_width=True)
    
    # Input box for the question
    question = st.text_input("Enter your question:")
    
    # Voice command button
    if st.button("Use Voice Command"):
        question = get_voice_command()

    if question:
        st.write(f"Question: {question}")
        answer = blip_vqa.get_answer(image, question)
        st.write(f"Answer: {answer}")
    else:
        st.write("Please enter a question or use the voice command.")

else:
    st.write("Please upload an image.")
