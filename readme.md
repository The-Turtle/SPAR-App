# SPAR Application Take-Home Task

> Topic: Exploring dangerous hidden reasoning capabilities of language models
>
> Mentor: Rohan Subramani
>
> Prompt: Use a Runpod GPU to finetune a huggingface transformer language model of your choice on a dataset of your choice, then evaluate the effect of the finetuning by running some of the same prompts on the pre- and post-finetuning models.

## Overview

This repository fine-tunes DistilGPT-2 on text datasets. The datasets included in this repository are:

1. The *Harry Potter* series, by J.K. Rowling
2. *Finnegans Wake*, by James Joyce
3. Shakespeare's complete works
4. Trump's social media posts (Twitter and Truth Social)

## Usage

1. Clone the repository:
```bash
git clone https://github.com/The-Turtle/SPAR-App.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the main script:
```bash
python main.py
```

The script will:
1. Prompt you to select a dataset
2. Train a model on the selected dataset, or load the model if it already exists
3. If training, produce a plot of the training and evaluation losses over epochs
4. Allow interactive text generation with custom prompts

To train on your own dataset, simply follow these steps:
1. Place your text file in the `data/` directory.
2. Run: `python main.py`.
3. Specify the name of your dataset when prompted.

## Training Loss Plots

Below are the training and evaluation loss plots for each dataset.

**Harry Potter** | **Joyce**
---|---
![Harry Potter Loss Plot](plots/harrypotter.png) | ![Joyce Loss Plot](plots/joyce.png)

**Shakespeare** | **Trump**
---|---
![Shakespeare Loss Plot](plots/shakespeare.png) | ![Trump Loss Plot](plots/trump.png)

With the exception of Finnegans Wake, all models showed convergence with decreasing training and evaluation losses across epochs. The evaluation losses start to taper off after around six epochs while the training losses continue to decrease, indicating that the model could be overfitted if trained for more epochs. The loss plots are saved in the `plots/` directory.

Sorting the models by their evaluation loss gives a rough estimate for the predictability of each dataset. Shakespeare's works are the most predictable, followed by the *Harry Potter series*, Trump's social media posts, and *Finnegans Wake*.

### Sample Output

Newlines were omitted and outputs were truncated at punctuation marks, if possible.

| Model | "The magic wand " | "riverrun past " | "To be or not to be " | "America is " |
|-------|---------------------|--------------|------------------|-----------------|
| base model | TODO | TODO | TODO | TODO |
| **Harry Potter** | The magic wand was broken. It was a fake; it was not, it was not — but, in fact, the wand that was the most powerful wand in the world, and the one that gave him power. | riverrun past the front door. “What happened?” he asked.<br>“Well, I think we should go back to the castle and go back to the school,” said Dumbledore, looking into the fire. “The house-elves were expelled, and I'm sure they'd have to wait to see me.” | To be or not to be not,” said Dumbledore. “We had to go. I didn’t want to be in Dumbledore’s house and not to get in with his friends, but to stay behind. And so I went. And I made sure I didn’t slip into the Pensieve, and then you slipped into the Pensieve and got in and out of here.” | America is the last refuge of our kind,” said Mr. Weasley, “that I want to see a bit of the Dark Lord come and visit me, perhaps.”<br>Mrs. Weasley nodded. She strode away toward the desk, which was full of portraits of the Minister of Magic. |
| **Joyce** | The magic wand from the wailers, O the mighty of them, and his wailers, O the mighty of them, and his wailers, O the mighty of them, and his wailers, O the mighty of them, and his wailers, | riverrun past the gates of our pilgrims and his psalmades, with the wailing ollum’s of our pewing of the spire’s of the spire’s of the spire’s of the spire’s of | To be or not to be in the dark of the night, but not to be in the darkness of the night, but not to be in the darkness of the night, but not to be in the darkness of the night, but not to be in the darkness of the night, but not to be in the darkness of the night, | America is that is, the wanstrawd for the sib and the russ. O the pape of the sib, the russ. O the trow, the sib. O the trow, the sib. O the hick. O the hiker.<br>—Pap!<br>—Hooligan’s bawdy! |
| **Shakespeare** | The magic wand<br>Exeunt.<br><br><br>
SCENE IV. The prison<br>Enter MARIANA and ANTON | riverrun past his.<br>And that he is afeard! I can tell you what he's that says,<br>And, like an honest man, I'll make you swear you know him,<br>And swear it is true, if it be true.<br>>FIRST GENTLEMAN. I know him well, sir.<br>He loves your lady dearly; and yet the man I am now | To be or not to be not-a-little,<br>I would not have been here.<br><br>Enter a MESSENGER<br><br>Here comes my master. Come, sir, you may do me good.<br>SECOND LORD. By my troth, my lord, sir, I am not to be well.<br>MESSENGER. I am a soldier | America is the best man in the world.<br>If he will be as fair as he is,<br>I'll make a fair offer, and have good terms with you;<br>And then, in good terms, I'll make a fair offer,<br>And have good terms with you.<br>PRINCESS OF FRANCE. My lords, be patient; if not,<br>And if not, you must be a very gentleman |
| **Trump** | The magic wand is to create the perfect app for your day job.<br>"""@sassy_g_welch: @realDonaldTrump I am going to run for President, I'm going to vote for you, I'm going to #MakeAmericaGreatAgain""" | riverrun past  on the water!<br>The new @NWS @WSJ poll finds that only 2% of those polled said the economy is strong. | To be or not to be very respectful to other people’s interests, we will always stand up for your rights. We’re not going to tolerate anything that doesn’t meet our basic basic standards of conduct. It’s not going to be tolerated by any nation or country. | America is looking at it like a very bad movie, just watch! My @foxandfriends interview discussing the economy, how much the military has lost since our invasion of Iraq |
