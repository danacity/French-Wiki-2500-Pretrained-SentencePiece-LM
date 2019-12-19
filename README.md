# French-Wiki-2500-Pretrained-SentencePiece-LM
I created this French SentencePiece Language Model, by using french Wikipedia articles that had more than 2500 words. The model is an AWD_LSTM, and it was trined for around 14 hours on a GCP using a v100 instance. If your goal is to create a language model more training would be necessary, and most likely a different architecture, but if you are using it as part of a classification task, you can fine-tune this model with your data set.   

If you would like to download the pretrained french language model is is available in my [google drive](https://drive.google.com/drive/folders/1hmlxzWQ2DRxAAR_Cdrm32OkwdkEgdF3J?usp=sharing).  
> Inside you will find several different files:  
> - fr_spm.ipynb - notebook I used to create the language spm model  
> - learner_fr_spm_enc.pth - encoder   
> - learner_mod_fr_spm.pkl - language model learner  
> - learner_mod_fr_spm_export.pkl - language model learner(using export)
> - learner_mod_fr_spm_save.pkl.pth - language model learner(using save)
> - learner_vocab_fr_spm.pkl - language model vocab
> - spm.model - Sentence Piece model(spm)
> - spm.vocab - Sentence Piece vocabulary

If you would like to use these models, I would recommend that you check out the amazing [Fast.ai NLP course](https://www.fast.ai/2019/07/08/fastai-nlp/), [github page](https://github.com/fastai/course-nlp), and the associated [turkish/spm notebook](https://github.com/fastai/course-nlp/blob/master/nn-turkish.ipynb)  

If you are following along with the [nn-turkish.ipynb notebook](https://github.com/fastai/course-nlp/blob/master/nn-turkish.ipynb), you will need to create a folder called ```tmp``` and put ```spm.model``` and ```spm.vocab``` inside of that ```tmp``` folder and In the step where you are creating a finetuned language model:
```
data_lm = (TextList.from_df(df, path_clas, cols='text', processor=SPProcessor.load(dest))
    .split_by_rand_pct(0.1, seed=42)
    .label_for_lm()           
    .databunch(bs=bs, num_workers=1))

data_lm.save(f'{lang}_clas_databunch')
```
You are going to define dest as the location where you put the ```tmp``` folder.

If you are using ```.save```, you can add an optional argument ```return_path=True``` so you can know where everything is being stored, since I found that it is not always abundantly clear.    
