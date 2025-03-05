from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams

class InferlessPythonModel:
    def initialize(self):        
        model_id = "unsloth/phi-4-GGUF"
        model_path = snapshot_download(repo_id=model_id,allow_patterns=["phi-4-Q4_K_M.gguf"])
        self.llm = LLM(model=f"{model_path}/phi-4-Q4_K_M.gguf",tokenizer="microsoft/phi-4")
        
    def infer(self, inputs):
        prompt = inputs["prompt"]
        system_prompt = inputs.get("system_prompt","You are a friendly bot.")
        temperature = inputs.get("temperature",1.0)
        top_p = inputs.get("top_p",1.0)
        top_k = int(inputs.get("top_k",-1))
        max_tokens = inputs.get("max_tokens",256)
        
        sampling_params = SamplingParams(temperature=temperature,top_p=top_p,
                                         top_k=top_k,max_tokens=max_tokens
                                        )
        conversation = [
           {
              "role": "system",
              "content": system_prompt
           },
           {
              "role": "user",
              "content": prompt
           }
        ]
        outputs = self.llm.chat(conversation, sampling_params)
        result_output = [output.outputs[0].text for output in outputs]
        return {"generated_text":result_output[0]}

    def finalize(self):
        self.llm = None
