from flask import Flask, request,  render_template
import torch
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
import pandas as pd


class CFG:
    model_name_or_path = 'cl-tohoku/bert-base-japanese'
    train_path = './sample_data/train.csv'
    val_path = './sample_data/val.csv'
    total_max_len = 512
    batch_size = 8
    accumulation_steps = 4
    epochs = 15
    debug = False


class chABSADataset(Dataset):

    def __init__(self, df, model_name_or_path, total_max_len):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row['text'],
            None,
            add_special_tokens=True,
            max_length=self.total_max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )

        ids = inputs['input_ids']
        ids = ids[:self.total_max_len]
        if len(ids) != self.total_max_len:
            ids = ids + [self.tokenizer.pad_token_id, ] * \
                (self.total_max_len - len(ids))
        ids = torch.LongTensor(ids)

        mask = inputs['attention_mask']
        mask = mask[:self.total_max_len]
        if len(mask) != self.total_max_len:
            mask = mask + [self.tokenizer.pad_token_id, ] * \
                (self.total_max_len - len(mask))
        mask = torch.LongTensor(mask)

        target = torch.LongTensor([row['target']])
        target = torch.nn.functional.one_hot(target, num_classes=2).squeeze()
        target = target.to(torch.float64)

        assert len(ids) == self.total_max_len

        return ids, mask, target

    def __len__(self):
        return self.df.shape[0]


class SM_Model(nn.Module):
    def __init__(self, model_path):
        super(SM_Model, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(
            model_path, output_hidden_states=True)

        self.fc_dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.config.hidden_size, 1)

        self._init_weights(self.fc)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        ids, mask = inputs
        outputs = self.model(ids, mask)
        last_hidden_states = outputs[0]
        prob = self.linear_relu_stack(last_hidden_states)
        prob = torch.mean(prob, dim=1).squeeze()
        return prob


def read_data(data):

    device = torch.device('cpu')
    return (data[0].to(device), data[1].to(device)), data[2].to(device)


args = CFG

model1 = SM_Model('cl-tohoku/bert-base-japanese')


# checkpoint = torch.load('/content/drive/MyDrive/Colab Notebooks/datasets/chabsa/outputs/model.pth')
model1.load_state_dict(torch.load(
    'model.pth', map_location=torch.device('cpu')))
model1.eval()


# Define the Flask app
app = Flask(__name__)

# Define the home page route


@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form
    text = request.form['text']

    # Tokenize and pad the input sequence

    # Make the prediction

    data = {'Id': [1],
            'text': [text],
            'target': [0]}

    test_df = pd.DataFrame(data)

    test_ds = chABSADataset(
        test_df, model_name_or_path=args.model_name_or_path, total_max_len=args.total_max_len)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    tbar = tqdm(test_loader, file=sys.stdout)

    preds = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model1(inputs)

                te_pred = pred.unsqueeze(dim=0)
                o_target = torch.argmax(te_pred, dim=1)
                preds.append(o_target)

    # will predict 0 or 1
    # Convert 'Long' tensor to floating-point tensor
    preds_float = preds[0].float()
    rounded_value = preds_float.round().detach().cpu().numpy().ravel()[0]
    print("rounded value: ", rounded_value)

    # Return the prediction as a string
    if rounded_value == 1.0:
        sentiment = 'POSITIVE'
    else:
        sentiment = 'NEGATIVE'

    return render_template('result.html', sentiment=sentiment)


if __name__ == '__main__':
    app.run(debug=True)
