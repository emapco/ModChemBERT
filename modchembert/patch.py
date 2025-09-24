import logging
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from deepchem.models.optimizers import LearningRateSchedule
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from deepchem.utils.typing import LossFn, OneOrMany
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForUniversalSegmentation,
)

logger = logging.getLogger(__name__)


def patch_deep_chem_hf_model():
    class HuggingFaceModelOverride(HuggingFaceModel):
        def load_from_pretrained(  # type: ignore
            self, model_dir: str | None = None, from_hf_checkpoint: bool = False, kwargs: dict | None = None
        ):
            """Load HuggingFace model from a pretrained checkpoint.

            The utility can be used for loading a model from a checkpoint.
            Given `model_dir`, it checks for existing checkpoint in the directory.
            If a checkpoint exists, the models state is loaded from the checkpoint.

            If the option `from_hf_checkpoint` is set as True, then it loads a pretrained
            model using HuggingFace models `from_pretrained` method. This option
            interprets model_dir as a model id of a pretrained model hosted inside a model repo
            on huggingface.co or path to directory containing model weights saved using `save_pretrained`
            method of a HuggingFace model.

            Parameter
            ----------
            model_dir: str
                Directory containing model checkpoint
            from_hf_checkpoint: bool, default False
                Loads a pretrained model from HuggingFace checkpoint.

            Note
            ----
            1. Use `load_from_pretrained` method only to load a pretrained model - a
                model trained on a different task like Masked Language Modeling or
                Multitask Regression. To `restore` a model, use the `restore` method.

            2. A pretrain model has different number of target tasks for pretraining and a finetune
                model has different number of target tasks for finetuning. Thus, they both have different
                number of projection outputs in the last layer. To avoid a mismatch
                in the weights of the output projection layer (last layer) between
                the pretrain model and current model, we delete the projection
                layers weights.
            """
            if kwargs is None:
                kwargs = {}
            if model_dir is None:
                model_dir = self.model_dir

            if from_hf_checkpoint:
                # FIXME Transformers library has an api like AutoModel.from_pretrained. It allows to
                # initialise and create a model instance directly without requiring a class instance initialization step
                # To use `load_from_pretrained` in DeepChem, we need to follow a two step process
                # of initialization class instance and then loading weights via `load_from_pretrained`.
                if self.task == "mlm":
                    self.model = AutoModelForMaskedLM.from_pretrained(
                        model_dir,
                        trust_remote_code=True,
                        config=self.config,
                        device_map=self.device,
                        ignore_mismatched_sizes=True,
                        **kwargs,
                    )
                elif self.task in ["mtr", "regression", "classification"]:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_dir,
                        trust_remote_code=True,
                        config=self.config,
                        device_map=self.device,
                        ignore_mismatched_sizes=True,
                        **kwargs,
                    )
                elif self.task == "universal_segmentation":
                    self.model = AutoModelForUniversalSegmentation.from_pretrained(
                        model_dir,
                        trust_remote_code=True,
                        config=self.config,
                        device_map=self.device,
                        ignore_mismatched_sizes=True,
                        **kwargs,
                    )
                else:
                    self.model = AutoModel.from_pretrained(
                        model_dir,
                        trust_remote_code=True,
                        config=self.config,
                        device_map=self.device,
                        ignore_mismatched_sizes=True,
                        **kwargs,
                    )
            elif not from_hf_checkpoint:
                checkpoints = sorted(self.get_checkpoints(model_dir))
                if len(checkpoints) == 0:
                    raise ValueError("No checkpoint found")
                else:
                    checkpoint = checkpoints[0]
                    data = torch.load(checkpoint, map_location=self.device)
                    # Delete keys of output projection layer (last layer) as the number of
                    # tasks (projections) in pretrain model and the current model
                    # might vary.

                    # When using Distributed Data Parallel (DDP) for training models, PyTorch automatically
                    # wraps model parameters in a module. prefix. This can cause issues when loading or
                    # saving model states because the key names in state_dict differ from their original
                    # single-GPU counterparts. To address this, model_state_dict is updated by removing
                    # the "module." prefix when saving or loading models.

                    data["model_state_dict"] = {
                        key.replace("module.", ""): value for key, value in data["model_state_dict"].items()
                    }
                    keys = data["model_state_dict"].keys()
                    if "classifier.out_proj.weight" in keys:
                        del data["model_state_dict"]["classifier.out_proj.weight"]
                    if "classifier.out_proj.bias" in keys:
                        del data["model_state_dict"]["classifier.out_proj.bias"]
                    if "classifier.dense.bias" in keys:
                        del data["model_state_dict"]["classifier.dense.bias"]
                    if "classifier.dense.weight" in keys:
                        del data["model_state_dict"]["classifier.dense.weight"]
                    self.model.load_state_dict(data["model_state_dict"], strict=False)

        def fit_generator(
            self,
            generator: Iterable[tuple[Any, Any, Any]],
            max_checkpoints_to_keep: int = 5,
            checkpoint_interval: int = 1000,
            restore: bool = False,
            variables: list[torch.nn.Parameter] | torch.nn.ParameterList | None = None,
            loss: LossFn | None = None,
            callbacks: Callable | list[Callable] | None = None,
            all_losses: list[float] | None = None,
        ) -> float:
            """Train this model on data from a generator.

            Parameters
            ----------
            generator: generator
                this should generate batches, each represented as a tuple of the form
                (inputs, labels, weights).
            max_checkpoints_to_keep: int
                the maximum number of checkpoints to keep.  Older checkpoints are discarded.
            checkpoint_interval: int
                the frequency at which to write checkpoints, measured in training steps.
                Set this to 0 to disable automatic checkpointing.
            restore: bool
                if True, restore the model from the most recent checkpoint and continue training
                from there.  If False, retrain the model from scratch.
            variables: list of torch.nn.Parameter
                the variables to train.  If None (the default), all trainable variables in
                the model are used.
            loss: function
                a function of the form f(outputs, labels, weights) that computes the loss
                for each batch.  If None (the default), the model's standard loss function
                is used.
            callbacks: function or list of functions
                one or more functions of the form f(model, step, **kwargs) that will be invoked
                after every step.  This can be used to perform validation, logging, etc.
            all_losses: Optional[List[float]], optional (default None)
                If specified, all logged losses are appended into this list. Note that
                you can call `fit()` repeatedly with the same list and losses will
                continue to be appended.

            Returns
            -------
            The average loss over the most recent checkpoint interval

            Note
            ----
            A HuggingFace model can return embeddings (last hidden state), attentions.
            Support must be added to return the embeddings to the user, so that it can
            be used for other downstream applications.
            """
            if callbacks is None:
                callbacks = []
            if not isinstance(callbacks, Sequence):
                callbacks = [callbacks]
            self._ensure_built()
            self.model.train()
            avg_loss = 0.0
            last_avg_loss = 0.0
            averaged_batches = 0
            if variables is None:
                optimizer = self._pytorch_optimizer
                lr_schedule = self._lr_schedule
            else:
                var_key = tuple(variables)
                if var_key in self._optimizer_for_vars:
                    optimizer, lr_schedule = self._optimizer_for_vars[var_key]
                else:
                    optimizer = self.optimizer._create_pytorch_optimizer(variables)
                    if isinstance(self.optimizer.learning_rate, LearningRateSchedule):
                        lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(optimizer)
                    else:
                        lr_schedule = None
                    self._optimizer_for_vars[var_key] = (optimizer, lr_schedule)
            time1 = time.time()

            # Main training loop.
            current_step = None
            for batch in generator:
                if restore:
                    self.restore()
                    restore = False
                inputs: OneOrMany[torch.Tensor]
                inputs, labels, weights = self._prepare_batch(batch)  # type: ignore

                optimizer.zero_grad()
                outputs = self.model(**inputs)  # type: ignore

                batch_loss = outputs.get("loss")
                batch_loss.backward()
                optimizer.step()
                if lr_schedule is not None:
                    lr_schedule.step()
                self._global_step += 1
                current_step = self._global_step

                # Detach to address the following warning:
                # UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
                avg_loss += batch_loss.detach()

                # Report progress and write checkpoints.
                averaged_batches += 1
                should_log = current_step % self.log_frequency == 0
                if should_log:
                    avg_loss = float(avg_loss) / averaged_batches
                    logger.info("Ending global_step %d: Average loss %g" % (current_step, avg_loss))  # noqa: UP031
                    if all_losses is not None:
                        all_losses.append(avg_loss)
                    # Capture the last avg_loss in case of return since we're resetting to 0 now
                    last_avg_loss = avg_loss
                    avg_loss = 0.0
                    averaged_batches = 0

                if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
                    self.save_checkpoint(max_checkpoints_to_keep)
                for c in callbacks:
                    try:
                        # NOTE In DeepChem > 2.8.0, callback signature is updated to allow
                        # variable arguments.
                        c(self, current_step, iteration_loss=batch_loss)
                    except TypeError:
                        # DeepChem <= 2.8.0, the callback should have this signature.
                        c(self, current_step)
                if self.tensorboard and should_log:
                    self._log_scalar_to_tensorboard("loss", batch_loss, current_step)
                if (self.wandb_logger is not None) and should_log:
                    all_data = dict({"train/loss": batch_loss})
                    self.wandb_logger.log_data(all_data, step=current_step)

            # Report final results.
            if averaged_batches > 0:
                avg_loss = float(avg_loss) / averaged_batches
                logger.info("Ending global_step %d: Average loss %g" % (current_step, avg_loss))  # noqa: UP031
                if all_losses is not None:
                    all_losses.append(avg_loss)
                last_avg_loss = avg_loss

            if checkpoint_interval > 0:
                self.save_checkpoint(max_checkpoints_to_keep)

            time2 = time.time()
            logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
            return last_avg_loss

    HuggingFaceModel.load_from_pretrained = HuggingFaceModelOverride.load_from_pretrained  # type: ignore
    HuggingFaceModel.fit_generator = HuggingFaceModelOverride.fit_generator  # type: ignore
