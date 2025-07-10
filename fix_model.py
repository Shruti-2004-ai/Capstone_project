import tensorflow as tf
from tensorflow.keras.layers import InputLayer

def fix_model(input_model_path, output_model_path):
    """
    Fixes a saved Keras model with incompatible InputLayer config and re-saves it.
    
    Args:
        input_model_path (str): Path to the problematic model (.h5 or .keras).
        output_model_path (str): Path to save the corrected model.
    """
    try:
        # Attempt to load the model with custom handling
        model = tf.keras.models.load_model(
            input_model_path,
            custom_objects=None,  # Add custom layers here if needed
            compile=False
        )
        
        # Rebuild the input layer if necessary (optional)
        if isinstance(model.layers[0], InputLayer):
            new_input = InputLayer(
                input_shape=model.layers[0].input_shape[1:],  # Extract shape from original
                name=model.layers[0].name,
                dtype=model.layers[0].dtype
            )
            model.layers[0] = new_input
        
        # Re-save the model in TF2 format
        model.save(output_model_path, save_format="tf")
        print(f"✅ Model fixed and saved to: {output_model_path}")
    
    except Exception as e:
        print(f"❌ Error fixing model: {str(e)}")

# Example usage
if __name__ == "__main__":
    fix_model(
        input_model_path="converted_model.keras",  # Problematic model
        output_model_path="fixed_model.keras"     # Corrected output
    )
