import logging
import tensorflow as tf
from cnn_classifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path
from cnn_classifier import logger


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        logging.info("getting vgg16 model as base model")
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top  # layers + (flatten or dense) 
        )
        logging.info(f"Saving base model with parameters input_shape={self.config.params_image_size},"
                     f"weights={self.config.params_weights}"
                     f", include_top={self.config.params_include_top}")
        self.save_model(path=self.config.base_model_path, model=self.model)
        logging.info(f"Base model saved to: {self.config.base_model_path}")

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            logging.info("freezing all layers from base model")
            for layer in model.layers:
                model.trainable = False  # This will freeze all the layers of the model
        elif (freeze_till is not None) and (freeze_till > 0):
            logging.info(f"frezzing layer of base model till -{freeze_till}")
            # As no. increases no. of layers from last will not frozen  
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        logging.info("Taking base model layers for full model")
        flatten_in = tf.keras.layers.Flatten()(model.output)
        logging.info("Adding Dense layer in last")
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )
        logging.info("Added Layers Sucessfully in full model")

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        logging.info("Compiling full model with optimizer = SGD ,"
                     f"loss = {tf.keras.losses.CategoricalCrossentropy()},"
                     "metrics = accuracy")

        full_model.summary(print_fn=logger.info)
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,  # Freeze all the layers
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        logging.info("Saving Full model")
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
        logging.info("Saved Full model in %s", self.config.updated_base_model_path)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
