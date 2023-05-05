import attr
import tensorflow as tf
import tensorflow_federated as tff

@attr.s(eq=False, frozen=True, slots=True)
class ClientOutput:
    """Sructure for outputs returned from clients during federated optimization.


    Attributes:
        weights_delta: A dictionary of updates to the model's trainable variables.
        client_weight: Wight to be used in a weighted mean when aggregating weights_delta.
        model_output: A structure matching """