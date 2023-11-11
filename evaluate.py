import os
import sys
import json
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoModelForCausalLM, AutoProcessor, pipeline

sys.path.append('/home/akshatgupta')
from textless.data.hubert_feature_reader import HubertFeatureReader
from textless.data.kmeans_quantizer import KMeansQuantizer
from textless.vocoders.hifigan.vocoder import CodeHiFiGANVocoder

def generate_with_offset(lm_model, input_ids, gen_len_ratio=5, temperature=0.8, do_sample=True, offset=None):
    if offset is None:
        offset = lm_model.config.offset

    input_len= int(input_ids.shape[-1])
    generation_len = int(min(250, gen_len_ratio * input_len))
    input_ids = input_ids.to(lm_model.device)
    generated_ids = lm_model.generate(offset + input_ids, max_length=generation_len, do_sample=True, temperature=temperature)

    return generated_ids - offset


def get_asr_pipeline():

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=1,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe


def get_models():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    #tokenization parameters
    ulm_path = '/home/akshatgupta/speech-prompting-setup/models/TWIST-350M'
    hubert_model_name = 'mhubert-base-25hz'
    quantizer_model_name = 'kmeans'
    vocab_size = 500
    hubert_tokenizer_path = "/home/akshatgupta/speech-prompting-setup/models/mhubert_base_25hz_cp_mls_cv_sp_fisher.pt"
    quantizer_path = '/home/akshatgupta/speech-prompting-setup/models/mhubert_base_25hz_cp_mls_cv_sp_fisher_L11_km500.bin'

    #load models
    tokenizer = HubertFeatureReader(hubert_tokenizer_path).to(device).eval()
    quantizer = KMeansQuantizer(quantizer_path)
    ulm_model = AutoModelForCausalLM.from_pretrained(ulm_path).to(device)
    vocoder = CodeHiFiGANVocoder.by_name(
            dense_model_name = hubert_model_name,
            quantizer_model_name = quantizer_model_name,
            vocab_size = vocab_size
        ).to(device).eval()

    return tokenizer, quantizer, ulm_model, vocoder, device


if __name__ == '__main__':
    do_sample = True
    temperature = 0.8
    prompt_word_len = 2
    gen_len_ratio = 8
    save_data = {}
    
    #define model locations here
    pipe = get_asr_pipeline()
    tokenizer, quantizer, ulm_model, vocoder, device = get_models()
    audio_path = 'test_audio2.flac'
    output_filename = 'test_output.json'

    #read audio
    audio, sample_rate = torchaudio.load(audio_path)
    audio = audio.squeeze(0).numpy()
    result = pipe(audio, batch_size=1, return_timestamps="word", generate_kwargs={"language": "english"})

    prompt_end_timestamp = result['chunks'][prompt_word_len-1]['timestamp'][1]
    prompt_text = ' '.join(word['text'].strip() for word in result['chunks'][:prompt_word_len])
    original_text = result['text'].strip()

    #create prompt audio
    prompt_audio = audio[:int(sample_rate * prompt_end_timestamp) + int(sample_rate/1000)]#adding 1 milisecond error window
    prompt_audio = torch.tensor(prompt_audio).reshape(1,-1).to(device)

    #tokenization
    output = tokenizer(prompt_audio)
    units = quantizer(output)
    units, durations = torch.unique_consecutive(units, return_counts=True)#deduplication step
    input_len = units.shape[0]

    #ulm generation
    input_ids = units.reshape(1, -1)
    generated_ids = generate_with_offset(ulm_model, input_ids, gen_len_ratio, temperature, do_sample)
    generated_ids = generated_ids[:, input_len:-1]


    #generate audio
    generated_audio = vocoder(generated_ids, dur_prediction = True)
    generated_audio = generated_audio.detach().cpu().numpy()
    
    #create generated audio transcript
    result = pipe(generated_audio, batch_size=1, return_timestamps="word", generate_kwargs={"language": "english"})
    generated_text = result['text'].strip()
    gen_text_len = len(result['chunks'])

    #save data
    save_data[audio_path] = {
        'prompt_word_len' : prompt_word_len,
        'full_text_transcript' : original_text,
        'prompt_transcript' : prompt_text,
        'generated_transcript' : generated_text,
        'generated_text_len' : gen_text_len,
        'generate_audio_len_ratio' : gen_len_ratio,
        'temperature' : temperature,
        'do_sample' : do_sample
    }

    with open(output_filename, "w") as outfile: 
        json.dump(save_data, outfile)
