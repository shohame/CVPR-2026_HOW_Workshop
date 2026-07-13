import torch
import PIL.Image
from nnsight import VisionLanguageModel
from workbench import WorkbenchServer, VLMVisualizer  # Example Workbench API components

# 1. Load your local Vision-Language Model
local_model_path = "models/llava-1.5-7b-hf"
llava = VisionLanguageModel(
    local_model_path, 
    dispatch=True, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# 2. Initialize the local Workbench server
app = WorkbenchServer(title="VLM Interpretability Workbench")

# 3. Register the Logit Lens tool with the Workbench backend
@app.register_tool(name="vlm_logit_lens", visualizer=VLMVisualizer)
def logit_lens_tool(prompt, image_path):
    # Load and convert the target image
    image = PIL.Image.open(image_path).convert("RGB")
    
    # Run the nnsight trace locally on your GPU
    with llava.trace(prompt, images=[image], remote=False) as tracer:
        top1_per_layer = list().save()
        
        # Intercept the residual stream at each layer
        for layer in llava.model.language_model.layers:
            # Apply final RMSNorm and lm_head linear projection
            logits = llava.lm_head(llava.model.language_model.norm(layer.output))
            
            # Extract the top-1 predicted token ID for each position
            top1_per_layer.append(logits.argmax(dim=-1))
            
    # Return the captured data to the React visualization widget
    return top1_per_layer

# 4. Start the local server
if __name__ == "__main__":
    print("Starting Workbench server. Open http://localhost:8080 in your browser.")
    app.run(host="localhost", port=8080)