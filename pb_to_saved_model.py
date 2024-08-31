# convert_pb_to_saved_model.py
import tensorflow as tf

def convert_pb_to_saved_model(pb_path, saved_model_path):
    # Disable eager execution
    tf.compat.v1.disable_eager_execution()

    # Load the .pb file
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Create a new graph and import the GraphDef
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    # Create a session from the graph
    with tf.compat.v1.Session(graph=graph) as sess:
        # Get input and output tensors
        input_tensor = graph.get_tensor_by_name("images:0")
        output_tensor = graph.get_tensor_by_name("features:0")

        # Create a concrete function
        @tf.function
        def serving_function(x):
            return output_tensor

        # Get the concrete function
        concrete_function = serving_function.get_concrete_function(
            tf.TensorSpec(input_tensor.shape, input_tensor.dtype)
        )

        # Save the model
        tf.saved_model.save(
            obj=sess,
            export_dir=saved_model_path,
            signatures={"serving_default": concrete_function}
        )

    print(f"Model saved to {saved_model_path}")

if __name__ == "__main__":
    pb_path = r"C:\Users\BQ Team 4\Documents\child\Therapist-and-Child-Detection-and-Tracking\mars-small128.pb"
    saved_model_path = r"C:\Users\BQ Team 4\Documents\child\Therapist-and-Child-Detection-and-Tracking\mars_saved_model"
    convert_pb_to_saved_model(pb_path, saved_model_path)