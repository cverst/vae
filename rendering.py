import matplotlib.pyplot as plt
import tensorflow as tf


def render_samples(dataset: tf.data.Dataset) -> None:
    """Render 8 examples of Animal Crossing villagers dataset and save to file.

    Args:
        dataset (tf.data.Dataset): Dataset from which to render villagers. Must be
            unbatched!
    """

    OUTPUT_FILENAME = "samples.jpg"

    index = 1

    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(wspace=0.5)

    for record in dataset.take(8):
        plt.subplot(2, 4, index)
        plt.axis("off")
        plt.imshow(record["image"].numpy())

        attributes = ", ".join(
            [
                record["gender"].numpy().decode("utf-8"),
                record["personality"].numpy().decode("utf-8"),
                record["species"].numpy().decode("utf-8"),
            ]
        )
        plt.title(record["name"].numpy().decode("utf-8") + "\n" + attributes)

        index += 1

    plt.close()

    fig.savefig(OUTPUT_FILENAME)


def render_loss(history: tf.keras.callbacks.History) -> None:
    """Render training and validation loss and save to file.

    Args:
        history (tf.keras.callbacks.History): Model training history for which to
            render loss.
    """

    OUTPUT_FILENAME = "loss.jpg"

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    
    ax.plot(history.history["loss"], label="Train", color="black")
    ax.plot(history.history["val_loss"], label="Validate", color="red")

    plt.title(label="Model loss by epoch", loc="center")
    ax.set(xlabel="Epoch", ylabel="Loss")
    plt.xticks(
        ticks=list(range(len(history.history["loss"]))),
        labels=list(range(1, len(history.history["loss"]) + 1)),
    )
    plt.legend()

    plt.close()

    fig.savefig(OUTPUT_FILENAME)
