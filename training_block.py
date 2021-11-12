import torch
from transformers import get_linear_schedule_with_warmup

class TrainingBlock():
    def __init__(self,epochs,weight_decay,lr,batch_size,device,eps =0.1):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
    
    def compute_metrics(self,pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        print(pred_ids.shape,pred_ids)
        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids)
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return wer

    def train(self,model,processor,train_dataloader,val_dataloader):
        self.model = model
        self.processor = processor

        print("start training")
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.2},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.eps)

        num_epochs = self.epochs
        total_steps = len(train_dataloader) * num_epochs
        print(total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        train_losses = []
        val_losses = []
        num_mb_train = len(train_dataloader)
        num_mb_val = len(val_dataloader)
        print(num_mb_train)

        if num_mb_val == 0:
            num_mb_val = 1

        training_status = []
        total_t0 = time.time()
        for epoch_i in range(0, num_epochs):

            #-------------------Training-----------------------
            print('Epoch {:} / {:}'.format(epoch_i + 1, num_epochs))

            t0 = time.time()
            total_train_loss = 0

            model.train()
            model.to(device)
            for step, batch in enumerate(train_dataloader):
                if step % 500 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                b_input_ids = batch[0].to(device)
                b_labels = batch[1].to(device)

                model.zero_grad()
        
                outputs = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
    
            avg_train_loss = total_train_loss / len(train_dataloader)             
            training_time = format_time(time.time() - t0)

            print("\n")
            print(" Average training loss: {0:.2f}".format(avg_train_loss))
            print(" Training epoch took: {:}".format(training_time))
                
            # ------------------Validation--------------------
            print("\n")
            print("Validation")

            t0 = time.time()
            model.eval()
            # model.to("cpu")
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            for batch in val_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                with torch.no_grad():        
                    outputs = model(b_input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits
                total_eval_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                total_eval_accuracy += flat_accuracy(logits, label_ids)
                
            avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            avg_val_loss = total_eval_loss / len(val_dataloader)
            
            val_time = format_time(time.time() - t0)
            
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(val_time))

            training_status.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': val_time
                }
            )

            output_dir = self.model_save + "model_" + str(epoch_i) + '_' + str(avg_val_accuracy)[:5]

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print("Saving model to %s" % output_dir)

            model_to_save = model.module if hasattr(model, 'module') else model 
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        print("\n")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
        