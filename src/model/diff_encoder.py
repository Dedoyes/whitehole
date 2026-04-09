import os.path as path
from transformers import BartTokenizer, BartForConditionalGeneration

encoder_path = path.abspath (__file__)

print (encoder_path)

tokenizer = BartTokenizer.from_pretrained ("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained ("facebook/bart-base")

print ("start")
inputs = tokenizer ("A fight with B, B win the match", return_tensors="pt")
print (inputs)

print (inputs["input_ids"].shape)
encoder_outputs = model.model.encoder (**inputs)
encoder_outputs_copy = encoder_outputs.last_hidden_state.clone ()

print (encoder_outputs.last_hidden_state.shape)

outputs = model.generate (
    encoder_outputs=encoder_outputs_copy,
    num_beams=1
)

print (encoder_outputs.last_hidden_state.shape)

print (tokenizer.decode (outputs[0]))
