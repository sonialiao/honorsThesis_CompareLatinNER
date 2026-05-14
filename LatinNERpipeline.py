# === Overhauled version of LatinNERpipeline.py from Ner-Latin-RANLP ===

from transformers import BatchEncoding
import subword_text_encoder as text_encoder
from transformers.pipelines.token_classification import TokenClassificationPipeline, AggregationStrategy
from typing import Optional, List, Tuple
import numpy as np

class TextEncoderWrapper:
	"""Wrapper to load tensor2tensor-style text encoder files"""
	def __init__(self, filepath):
		self.vocab = []
		self.string_to_id = {}
		with open(filepath, 'r', encoding='utf-8') as f:
			for idx, line in enumerate(f):
				token = line.rstrip('\n')
				self.vocab.append(token)
				self.string_to_id[token] = idx
	
	def get_piece_size(self):
		return len(self.vocab)
	
	def id_to_piece(self, idx):
		return self.vocab[idx] if idx < len(self.vocab) else '<unk>'
	
	def encode_as_ids(self, text):
		"""Simple greedy encoding (will fail for OOV, but matches original behavior)"""
		ids = []
		text_lower = text.lower() if isinstance(text, str) else text
		# For single tokens
		if text_lower in self.string_to_id:
			ids.append(self.string_to_id[text_lower])
		else:
			# Fallback: try character-level fallback or unknown
			ids.append(self.string_to_id.get('<unk>', 0))
		return ids

class LatinTokenizer():
	def __init__(self, encoder):
		self.vocab={}
		self.reverseVocab={}
		self.encoder=encoder

		self.vocab["[PAD]"]=0
		self.vocab["[UNK]"]=1
		self.vocab["[CLS]"]=2
		self.vocab["[SEP]"]=3
		self.vocab["[MASK]"]=4
		self.model_max_length=256
		self.is_fast=False


		self.cls_token_id = self.vocab["[CLS]"]
		self.pad_token_id = self.vocab["[PAD]"]
		self.sep_token_id = self.vocab["[SEP]"]
        
		for key in self.encoder._subtoken_string_to_id:
			self.vocab[key]=self.encoder._subtoken_string_to_id[key]+5
			self.reverseVocab[self.encoder._subtoken_string_to_id[key]+5]=key


	def convert_tokens_to_ids(self, tokens):
		wp_tokens=[]
		for token in tokens:
			if token == "[PAD]":
				wp_tokens.append(0)
			elif token == "[UNK]":
				wp_tokens.append(1)
			elif token == "[CLS]":
				wp_tokens.append(2)
			elif token == "[SEP]":
				wp_tokens.append(3)
			elif token == "[MASK]":
				wp_tokens.append(4)

			else:
				wp_tokens.append(self.vocab[token])

		return wp_tokens

	def tokenize(self, text, split_on_tokens=True):
		if split_on_tokens:
			tokens = [token.lower() if token not in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] else token for token in text]
		else: 
			tokens = [token.lower() for token in text.split()]

		wp_tokens=[] #word-piece tokens
		check = []

		for n, token in enumerate(tokens):
			# print(token)

			if token in {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}:
				wp_tokens.append(token)
				check.append(n)
			else:

				wp_toks=self.encoder.encode(token)

				for wp in wp_toks:
					wp_tokens.append(self.reverseVocab[wp+5])
					check.append(n)

		return wp_tokens, check
	
	def calculate_attention_masks(self, wp_tokens):
		attention_masks = []
		
		for token in wp_tokens:
			if token == self.pad_token_id:
				attention_masks.append(0)
			else:
				attention_masks.append(1)
				
		return attention_masks
	
	def pad_max_length_and_add_specials_tokens_also(self, tokens, wp_tokens):
		
		MAX_LENGTH = 256
		wp_tokens.insert(0, self.cls_token_id)
		tokens.insert(0, '[CLS]')
		wp_tokens.append(self.sep_token_id)
		tokens.append('[SEP]')
		
		if len(wp_tokens) > 256:
			wp_tokens = wp_tokens[:256]
		
		else:
			while len(wp_tokens) < 256:
				wp_tokens.append(self.pad_token_id)
				tokens.append('[PAD]')

		return tokens, wp_tokens
	
	def pad_max_length_and_add_specials(self, wp_tokens):

		MAX_LENGTH = 256
		wp_tokens.insert(0, self.cls_token_id)
		wp_tokens.append(self.sep_token_id)
		
		if len(wp_tokens) > 256:
			wp_tokens = wp_tokens[:256]
		
		else:
			while len(wp_tokens) < 256:
				wp_tokens.append(self.pad_token_id)

		return wp_tokens
	
	def decode_to_string(self, input_ids):
		tokens = [self.reverseVocab[x] for x in input_ids if x > 4]
		return "".join(tokens).replace('_', ' ')

	def save_pretrained(self, output_dir):
		pass


def extend_clear_list(temp, fixed, item):
    temp.append(item)
    fixed.append(int(np.mean(temp)))
    temp.clear()

def aggregate_ents(orig_tokens, wp_tokens, check, labels):
    try:
        assert len(wp_tokens) == len(labels) == len(check)
    except AssertionError:
        print('lenght tokens labels are not equal')
        print(wp_tokens)
        print(check)
        
    fixed_labels = []
    
    temp_label = []
    
    for i in range(len(wp_tokens)):
        try:
            if check[i+1] != check[i] and len(temp_label) == 0:
                fixed_labels.append(labels[i])
            
            elif check[i+1] != check[i]:
                extend_clear_list(temp_label, fixed_labels, labels[i])
            else:
                temp_label.append(labels[i])
        except IndexError:
            if len(temp_label) == 0:
                fixed_labels.append(labels[i])
            
            elif len(temp_label) != 0:
                extend_clear_list(temp_label, fixed_labels, labels[i])
            
    try:
        assert len(orig_tokens) == len(fixed_labels)
    except AssertionError:
        print('lenght of original tokens, aggregated predictions and labels are not equal')
        print(fixed_labels)
        print(check)
        print(orig_tokens)
    
    return fixed_labels

def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


# Load text encoder model
tokenizer = LatinTokenizer(text_encoder.SubwordTextEncoder('latin-bert/models/subword_tokenizer_latin/latin.subword.encoder'))

class LatinNerPipeline(TokenClassificationPipeline):

    # CHANGED: replaced **kwargs signature with the exact signature from
    # TokenClassificationPipeline in transformers 4.28.0, and delegated to
    # super() so the base class can handle its own parameter routing correctly.
    def _sanitize_parameters(
        self,
        ignore_labels=None,
        grouped_entities: Optional[bool] = None,
        ignore_subwords: Optional[bool] = None,
        aggregation_strategy: Optional[AggregationStrategy] = None,
        offset_mapping: Optional[List[Tuple[int, int]]] = None,
        stride: Optional[int] = None,
		split_on_words: bool = False,			# custom parameter
    ):
        preprocess_params, forward_params, postprocess_params = \
            super()._sanitize_parameters(
                ignore_labels=ignore_labels,
                grouped_entities=grouped_entities,
                ignore_subwords=ignore_subwords,
                aggregation_strategy=aggregation_strategy,
                offset_mapping=offset_mapping,
                stride=stride,
            )
        if split_on_words:
            preprocess_params["split_on_words"] = split_on_words
        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, sentence, offset_mapping=None, split_on_words=False, **preprocess_params):
        test = {}
        if split_on_words:
            tokens = sentence
        else:
            tokens = [token.lower() for token in sentence.split()]

        wp_tokens, check = tokenizer.tokenize(tokens)
        token_ids = tokenizer.convert_tokens_to_ids(wp_tokens)
        test['inputs'] = tokens
        test['input_ids'] = [token_ids]
        test['attention_mask'] = [tokenizer.calculate_attention_masks(token_ids)]
        test['wp_tokens'] = wp_tokens
        test['check'] = check

        test['is_last'] = True

        # CHANGED: `return test` -> `yield test`
        # In transformers 4.28+, the pipeline internals call next() on whatever
        # preprocess() returns, so it must be a generator, not a plain dict.
        yield test


    def _forward(self, model_inputs, **forward_params):
        input_data = BatchEncoding(
        {'input_ids': model_inputs['input_ids'], 
         'attention_mask': model_inputs['attention_mask']},
        tensor_type="pt"
    	)
        outputs = self.model(input_ids=input_data['input_ids'], attention_mask=input_data['attention_mask'])
        model_inputs['outputs'] = outputs
        return model_inputs

    def postprocess(self, all_outputs, aggregation_strategy=AggregationStrategy.NONE, ignore_labels=None):
        # CHANGED: removed the dead first line `logits = all_outputs['outputs']["logits"] ...`
        # which was immediately overwritten on the next line anyway.
        # CHANGED: all_outputs is now a list of dicts from the pipeline internals, grab the first element
        output = all_outputs[0]
        logits = output['outputs'].logits[0].detach().numpy()

        probabilities = [softmax(i) for i in logits]
        best_classes = [np.argmax(prob) for prob in probabilities]
        logits = logits.tolist()
        agg_classes = aggregate_ents(output['inputs'], output['wp_tokens'], output['check'], best_classes)
        labels = [self.model.config.id2label[best_class] for best_class in agg_classes]

        output['logits'] = logits
        output['labels'] = labels

        return output