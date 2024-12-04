Lab 3: Natural Language Processing with Deep Learning Models
Objective:
In this lab, I worked on implementing, fine-tuning, and evaluating deep learning models for Natural Language Processing (NLP) tasks. 
The focus was on using RNN, GRU, and LSTM architectures to process Arabic text data. Additionally, I fine-tuned the GPT-2 pre-trained model on a custom dataset and evaluated the models using various metrics.

Key Learnings:
Data Collection & Preprocessing:

Collected Arabic text data using BeautifulSoup from websites.
Applied text preprocessing techniques including tokenization, stemming, stopword removal, and lemmatization using libraries such as NLTK and pyarabic.
Converted the text data into numerical format suitable for training models.
Model Training:

Built and trained RNN, GRU, and LSTM models using PyTorch.
Fine-tuned GPT-2 using a custom dataset, adapting the pre-trained model for specific text generation tasks.
Evaluation:

Evaluated the models using standard metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) for regression tasks.
Used the BLEU score to evaluate text generation models, measuring how well the generated text matched the reference text.
Tracking and Experimentation:

Integrated Weights & Biases (W&B) for experiment tracking, visualization, and logging. This allowed me to monitor the training process in real-time and compare different runs.
Methodology:
Data Collection:

Used BeautifulSoup to scrape Arabic text data from selected websites.
Preprocessed the collected data using tokenization and cleaning techniques suitable for Arabic text.
Model Architecture:

RNN: Used for basic sequential data processing.
GRU: Used for improving performance in sequential tasks with gated recurrent units.
LSTM: Leveraged for capturing long-range dependencies in the text.
GPT-2 Fine-Tuning: Fine-tuned a pre-trained GPT-2 model on the custom dataset to generate more relevant text.
Training:

Implemented custom training loops using PyTorch and used Adam optimizer with a learning rate scheduler.
Applied early stopping techniques to avoid overfitting and to ensure the models generalized well on unseen data.
Evaluation:

MSE, MAE, and RMSE were used to evaluate the performance of regression models.
BLEU score was used to evaluate the quality of text generation from the models.
Logging with W&B:

Used Weights & Biases (W&B) to log model training metrics and hyperparameters for future reference and comparison.
Results:
Training Loss:
The models showed a significant reduction in training loss across multiple epochs, indicating that the models were learning effectively.

Evaluation Metrics:

MSE and MAE scores demonstrated a high level of accuracy in predicting text-related tasks, with the lowest loss values observed in the final epochs of training.
BLEU score for text generation tasks indicated that the GPT-2 model produced highly relevant and contextually appropriate outputs.

Brief Synthesis:
During the proposed lab, I gained hands-on experience in several key areas of machine learning and natural language processing (NLP). 
I began by learning how to collect and preprocess Arabic text data from websites using tools like BeautifulSoup. This included tokenization,
stemming, stopword removal, and other preprocessing techniques using libraries like NLTK and pyarabic. I then moved on to building and training neural network
models such as RNN, GRU, and LSTM using PyTorch. Through this, I gained practical experience in working with sequential data and learned how to fine-tune pre-trained
models like GPT-2 on a custom dataset. Additionally, I learned how to evaluate models using various metrics such as MSE, MAE, and BLEU score, and used tools like Weights & Biases
(W&B) for tracking experiments and visualizing training progress. This lab provided valuable exposure to NLP tasks, model fine-tuning, performance evaluation, and the use of deep
learning techniques to work with text data. It has equipped me with practical skills to apply in future NLP projects, including text generation, classification, and evaluation.

