from vllm import LLM, SamplingParams

class InferlessPythonModel:
    def initialize(self):
        self.llm = LLM(model="kaitchup/Phi-4-AutoRound-GPTQ-4bit", quantization="gptq")

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
