from transformers import AutoConfig
config = AutoConfig.from_pretrained("Nikeytas/videomae-crime-violence-detector")
print(config.id2label) 