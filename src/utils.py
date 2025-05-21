import re
import html
from bs4 import BeautifulSoup

def clean_for_bert(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    
    text = html.unescape(text)
    
    text = BeautifulSoup(text, "html.parser").get_text()
    
    text = re.sub(r'https?://\S+', '', text)
    
    text = re.sub(r'\b(http|href)\b', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'&\w+;', '', text)
    text = re.sub(r'[@#]\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

word_re = re.compile(r"\b[a-zA-Z]{2,}\b", re.IGNORECASE)

def extract_clean_words(text):
    return word_re.findall(text)


class EarlyStopper:
    def __init__(self, models, patience=5, min_delta=0.001, restore_best_weights=True, save_weights=False):
        self.models = models # list for model + tokenizer
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_weights = None
        self.restore_best_weights = restore_best_weights
        self.save_weights = save_weights

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = [copy.deepcopy(model.state_dict()) for model in self.models]

        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    for model, weights in zip(self.models, self.best_weights):
                        model.load_state_dict(weights)
                    if self.save_weights:
                        for model, weights in zip(self.models, self.best_weights):
                            model_name = model.__class__.__name__
                            torch.save(weights, f'{model_name}_best_weights.pt')
                            print(f"Weights for {model_name} have been saved")
                return True

        return False

# source: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch