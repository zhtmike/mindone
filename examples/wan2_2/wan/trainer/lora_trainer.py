import mindspore.nn as nn

from mindone.peft import LoraConfig, PeftModel, get_peft_model


class LoRATrainer:
    def __init__(self, pipeline, train_loader, val_loader, training_config, generation_config):
        self.pipeline = pipeline
        self.pipeline.model = self._configure_peft(self.pipeline.model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.training_config = training_config
        self.generation_config = generation_config

    def _configure_peft(self, model: nn.Cell) -> PeftModel:
        target_modules = ["q", "k", "v", "o"]
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        model: PeftModel = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def train_epoch(self):
        self.pipeline.model.set_train(True)

        for i, batch in enumerate(self.train_loader):
            if i % self.training_config.get("validation_interval", 100) == 0:
                self.validate()

            inputs, targets = batch

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

    def validate(self):
        self.pipeline.model.set_train(False)
        self.pipeline.generate(**self.generation_config)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
