#include <torch/torch.h>
#include <torch/optim/adamw.h>
#include <torch/utils/data/dataloader.h>

using namespace torch;

void train(Accelerator& accelerator, const Config& config) {
  // Set the random seed.
  set_seed(config['seed']);

  // Print the config.
  accelerator.print(config);
  accelerator.print(f"Using {accelerator.num_processes} CPUs");

  // Load the tokenizer.
  auto tokenizer = AutoTokenizer::from_pretrained(config['tokenizer_name'], model_max_length=config['max_length']);
  // If no pad token, set it to eos.
  if (tokenizer.pad_token is None) {
    tokenizer.pad_token = tokenizer.eos_token;
  }

  // Load the data.
  auto train_dataloader, val_dataloader = load_data(config, tokenizer);

  // Create the model.
  auto model = AutoModelForCausalLM.from_pretrained(config["model_name"], use_cache=False);
  if (config["lora"]) {
    auto peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    );
    model = get_peft_model(model, peft_config);
    model.print_trainable_parameters();
  }

  // Create the optimizer.
  auto optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]);

  // Create the scheduler.
  auto scheduler = get_scheduler(
      name="cosine",
      optimizer=optimizer,
      num_warmup_steps=config["warmup_steps"] * accelerator.num_processes,
      num_training_steps=config["total_training_steps"],
  );

  // Prepare the model, optimizer, and scheduler for training.
  model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
      model, optimizer, train_dataloader, val_dataloader, scheduler
  );

  // Train the model.
  for epoch in range(config["num_epochs"]) {
    // Train the model for one epoch.
    train_epoch(model, optimizer, train_dataloader, scheduler, config);

    // Evaluate the model on the validation set.
    auto val_loss = evaluate(model, val_dataloader);

    // Log the training and validation losses.
    accelerator.log({
      "train_loss": train_loss.compute(),
      "val_loss": val_loss.compute(),
    });

    // Save the model checkpoint.
    accelerator.save_state(f"{config['output_dir']}/epoch_{epoch}");
  }

  // Print the final loss.
  auto val_loss = evaluate(model, val_dataloader);
  accelerator.print(f"Final loss: {val_loss.compute()}");

  // Save the final model checkpoint.
  accelerator.save_state(f"{config['output_dir']}/final");
}
