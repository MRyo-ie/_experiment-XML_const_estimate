
from abc import ABCMeta, abstractmethod



class Seq2Seq_TokenizerABC(metaclass=ABCMeta):
    def __init__(self, input_lang, output_lang, device, tensor_type="pt"):
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.device = device
        self.tensor_type = tensor_type

    @abstractmethod
    def get_tensor_from_sentence(self, sentence, is_input: bool):
        raise NotImplementedError()

    @abstractmethod
    def get_tensors_from_pair(self, pair, tensor_type="pt"):
        raise NotImplementedError()



