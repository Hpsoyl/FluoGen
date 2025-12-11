accelerate launch --mixed_precision="fp16"  /data0/syhong/BioDiff/src/train_foundation_model.py \
  --resolution=512 \
  --output_dir="/data4/syhong_temp/BioDiff_model/wo_vpre" \
  --mixed_precision="fp16" \
  --train_batch_size=2 \
  --num_epochs=10 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --lr_warmup_steps=100 \
  --checkpointing_steps=5000 \
  --validation_steps=2500 \
  --prediction_type="epsilon" \
  --pretrained_model_name_or_path="/data0/syhong/stable-diffusion-v1-5" \
  --validation_prompts "F-actin of COS-7" \
                      "ER of COS-7" \
                      "CCPs of COS-7" \
                      "E.coli" \
                      "F-actin of BPAE" \
                      "nucleus of BPAE" \
                      "mitochondria of BPAE" \
                      "nuclei of hela" \
                      "Actin of HUVEC" \
                      "RNA of U2OS" \
                      "DNA of U2OS" \
                      "ER of U2OS" \
                      "Golgi and Actin of U2OS" \
                      "Tubulin of MCF-7" \
                      "mitochondria of U2OS" \
                      "" \
  --ddpm_timestep_spacing="trailing" \
  --lr_scheduler="linear" \
  --resume_from_checkpoint="/data4/syhong_temp/BioDiff_model/wo_vpre/checkpoint-5000" \
  