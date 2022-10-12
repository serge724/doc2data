import os
import numpy as np
import tensorflow as tf
from doc2data.experimental.utils import plot_keras_history
from doc2data.experimental.task_processors import PageRotationClsProcessor

try:
   import mlflow
except:
   pass


class KerasTrainer:
    """Class for training Keras models."""

    @staticmethod
    def train_model(
        training_params,
        model,
        data_processor,
        loss,
        metric,
        mlflow_tracking = False
    ):
        # start mlflow run
        if mlflow_tracking: mlflow.start_run()

        train_data = data_processor.initialize_split(data_split = 'train_set', batch_size = training_params['batch_size'])
        val_data = data_processor.initialize_split(data_split = 'val_set', batch_size = training_params['batch_size'])
        test_data = data_processor.initialize_split(data_split = 'test_set', batch_size = training_params['batch_size'])

        data_processor.train_set = train_data
        data_processor.val_set = val_data
        data_processor.test_set = test_data

        checkpointer = tf.keras.callbacks.ModelCheckpoint(
            'tmp_checkpoints/model.hdf5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True
        )

        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = training_params['learning_rate']),
            loss = loss,
            metrics = [metric]
        )

        history = model.fit(
            x = train_data,
            epochs = training_params['epochs'],
            validation_data = val_data,
            callbacks = [checkpointer],
            workers = 8,
            use_multiprocessing = True,
            verbose = 1
        )

        model.load_weights(checkpointer.filepath)

        # log results in mlflow
        if mlflow_tracking:
            mlflow.log_params(training_params)
            best_epoch = np.argmin(history.history['val_loss'])
            mlflow.log_metric('best_epoch', best_epoch)
            mlflow.log_metric('best_val_loss', history.history['val_loss'][best_epoch])
            mlflow.log_metric(f'best_val_{metric}', history.history[f'val_{metric}'][best_epoch])
            for key, values in history.history.items():
                for epoch, value in enumerate(values):
                    mlflow.log_metric(key, value, step = epoch)
            plot = plot_keras_history(history)
            mlflow.log_figure(plot, 'training_progress.png')
            os.makedirs('models', exist_ok = True)
            run = mlflow.active_run()
            model.save(os.path.join('models', '%s.h5'%run.info.run_id))
            test_loss, test_metric = model.evaluate(test_data)
            mlflow.log_metrics({'test_loss': test_loss, f'test_{metric}': test_metric})

            # end mlflow run
            mlflow.end_run()
