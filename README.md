#### Hi there, ####

It's 2020 and a crazy year goes to an end, not a good one. Machine learning is growing fast and breakthroughs have been made in several ways. 
2020 feels like the absolute best year for ML. One term that was frequently mentioned this year was artificial general intelligence (AGI).
With GPT-3, a new type of system was born that has undoubtedly had the greatest impact on the development of ML systems since its inception. The first AGI?
GPT-3 is not brand new, just the successor to GPT-2.

Let's see what GPT-2 says about GPT-3.\
GPT-3 is: ", well, the best ML system of the modern era. It has the best ML, it is the best ML, it is the most sophisticated ML, it is the best ML, it is the best ML, it is"

Ok, that was a pathetic attempt with GPT-2 on [Huggingface](https://transformer.huggingface.co/doc/distil-gpt2).
The point is, there is this theory, the ultimate approximation theory for neural networks.
And this system with its developers opened a door and showed possibilities that were previously unimaginable. Neural networks that I work with can take numbers as data and make some simple kind of prediction, but GPT-3 uses words. 
My neural networks are limited in one form. But words are not, anything can be a word, my data or every color pixel in an image, even an application or website can be words. It's not just a prediction system, it's a pattern generator in possibly the most nifty way 2020.
<p align="center">
  <img src="https://raw.githubusercontent.com/grensen/ML_2020/main/figures/gpt-3_website_google.png" width="542" height="337" />
</p>

A description of the Google website is enough and GPT-3 creates the result with accompanying code, mind blowing.\
[GPT-3 Demo: New AI Algorithm Changes How We Interact With Technology](https://www.youtube.com/watch?v=8V20HkoiNtc).  

When I started AI, I was talking to someone about AI and all that stuff. He told me he wouldn't waste his time on AI until those systems can build games. This could be possible now, with a lot more work. [GPT3: An Even Bigger Language Model - Computerphile](https://www.youtube.com/watch?v=_8yVOC4ciXc).\
Unfortunately I don't work with NLP yet. My work deals with technical aspects of creating deep neural networks.

---

Do you know Wolfram Alpha, this mystical website? The founder of this site is Stephen Wolfram, he deals with a new fundamental theory of physics. [Stephen Wolfram: Fundamental Theory of Physics, Life, and the Universe | Lex Fridman Podcast #124](https://www.youtube.com/watch?v=-t1_ffaFXao&feature=youtu.be&t=560).\
*"Time is the progression of computation"*

On the other side, [Werner Heisenberg](https://www.youtube.com/watch?v=THVmJpXQL9g), who is said to have developed the world formula, did not believe that there was a global theory.\
[Max Planck](https://www.youtube.com/watch?v=IQPPqagR-kQSadly) would probably say something like: "The luck of the scientist is not to own a truth, but to fight for the truth." This seems to me not a good translation.
Despite I don't really have a deep understanding of all that, it seems related.

Don't forget to mention the little [Google outbreak](https://www.youtube.com/watch?v=B9PL__gVxLI) in 2020, thanks Yannic Kilcher for another brilliant explanation.\
With Demon, a new promising optimizer has also been released. [Demon: Momentum Decay for Improved Neural Network Training](https://arxiv.org/abs/1910.04952).\
And faster training with less data can be reached with [Faster Neural Network Training with Data Echoing (Paper Explained)](https://www.youtube.com/watch?v=bFn2xcGi1TQ). 

What do you think of the term [Shortcut Learning](https://www.youtube.com/watch?v=D-eg7k8YSfs)? It describes a wrong learning path in neural networks.\
My first thought after the term "shortcut learning" led me to the idea to take the activation level of a MNIST sample divided by the neurons of that sample e.g. 160 / 784 = 0.2.
The next 4 neurons did the same for the upper left, upper right, lower left and lower right parts of the sample, all values are between 0 and 1. With these 5 input neurons instead of 784 an accuracy of > 50% was possible, unexpectedly. What will 16+1 inputs predict?

---

As icing on the cake, Microsoft has released NET 5 with good [arguments](https://devblogs.microsoft.com/dotnet/performance-improvements-in-net-5/).
Since all my ML work in 2020 is based on C#, this release is definitely an ML highlight for me. This leads me to goodgame (gg), my main work for 2020 and the interface for all my future work.\
goodgame feels great and is based on my 2019 work, the [perceptron concept](https://github.com/grensen/perceptron_concept).
I am really proud of this result and it is my attempt to describe a hermeneutic circle for neural networks. 

What exactly is goodgame? I would describe the experience like an arcade game to play neural networks on a high abstraction level.
This is to help demystify this pattern recognizer that we know as a deep neural network.

The NET 5 version of [goodgame_copy.cs](https://raw.githubusercontent.com/grensen/ML_2020/main/goodgame_copy.cs) is the finest. A fast way to run the app is to [download](https://drive.google.com/file/d/18yf7niFkdQKvt96rk5A1pcWo5iqUj3Fq/view?usp=sharing) and extract the goodgame folder to your C: directory, start gg.  [More about goodgame and how to run with Visual Studio](https://github.com/grensen/gif_test) and [How to activate NET 5 in VS 2019](https://stackoverflow.com/questions/60843091/net-5-is-not-available-in-visual-studio-2019) – I like the new top level code feature the most.

What tells you this picture? 
<p align="center">
  <img src="https://raw.githubusercontent.com/grensen/ML_2020/main/figures/new_connection_jiw.jpg" width="320" height="134" />
</p>

[Neuroscientist Manfred Spitzer](https://www.youtube.com/watch?v=fA97Nr4UxsY&feature=youtu.be&t=511) said this is the most important finding in the last half century of neuroscience. 
Even if neural networks are not our brain, the idea was sounding so interesting, that I put it to the test. 
For an algorithm, I have interpreted this as frequent use of the neurons leads to an increase in new connections. The neural network grows, in this case a whole node.
To make sure that this works, an test must show that the copy of a neuron keeps the same values as the source. 
You can see it on the last neuron on the first hidden layer, it's just the copy of the most activated neuron, here with a value of 11.131. 

![ggcopy](https://raw.githubusercontent.com/grensen/ML_2020/main/figures/ggcopy.png)

With goodgame_copy.cs this function is available. Ok, it doesn't seem to make sense for the training algorithm to use only one copy of a neuron, probably we would add some noise. 
But as it turns out, playing with different copy nodes for testing increases the overall accuracy from 93.29% to 93.54%, as shown in the figure. Instead of calculating two nodes, the source + the copy of it, we could simply multiply the well-trained weights of that single node by 2. 
Simply multiplying all the weights on a node to increase accuracy would be a much faster optimization than the entire backpropagation process. This seems like a valuable area of research. 

Efficiency is a key gg aspect, also surprising to me, dropout can speed up training.\
Another neat optimization to speed up training for output neurons is this [ReLUmax technique](https://github.com/grensen/ReLUmax).

Autoencoder are hot and gg is ready for that, here is an impression.

![ggae](https://raw.githubusercontent.com/grensen/ML_2020/main/figures/ggauto.png)

This autoencoder works with a neural network. Autoencoders are a hot topic, as the example slightly shows. 

Another idea to use gg is to create datasets to make predictions. The [PureAI Editors](https://pureai.com/articles/2020/11/05/ml-biology-images.aspx) explain the RxRx1 dataset really well.
The dataset can be found [here](https://www.rxrx.ai/rxrx1#Download), but with 512 * 512 pixels for each sample, a way to handle the RxRx1 data was needed.

![ggrx](https://raw.githubusercontent.com/grensen/ML_2020/main/figures/ggrxrx1.png)

The red rectangles show the attention areas from the source to the crop dimension. Each of the 5 shrinked and sharpened samples represents a part. 
Unfortunately, the labels for the RxRx1 data are not available now, so predictions have to wait.

---

For a serious approach with images and especially from the RxRx1 data is a convolutional neural network needed. 
The convolution algorithm for the first layer is based on the paper [Derivation of Backpropagation in Convolutional Neural Network (CNN) Zhifei Zhang](http://web.eecs.utk.edu/~zzhang61/docs/reports/2016.10%20-%20Derivation%20of%20Backpropagation%20in%20Convolutional%20Neural%20Network%20(CNN).pdf). 
<p align="center">
  <img src="https://raw.githubusercontent.com/grensen/ML_2020/main/figures/zhang_cnn_paper.png" />
</p>

The heart is replaced with the perceptron concept, the first convolution layer is just an additional system that acts as a feature extractor for the deep neural network.

![ggcnn](https://github.com/grensen/ML_2020/blob/main/figures/ggcnn.png)

With a training accuracy of almost 95%, this model clearly outperforms a deep neural network with the same size of hidden neurons. 
If you carefully reconstruct the 6 convolutions to the source, things might start to make sense.

---

The ICLR paper of the year 2019, [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635), was about pruning. 
Pruning means usually the drop of a particular weight out of the network. That can also be called unstructured pruning or weight pruning. 
Structured pruning means pruning an entire node with its associated weights (very efficient and already implemented in goodgame). 
With GPT-3 and the Transformer architecture, pruning techniques seem to expand as well.

There are two fundamental ways to implement unstructured pruning. Soft pruning uses an ordinary network with a bool array for the weights or sets the pruned weight to zero and prevents updates so that the weight stops working. Hard pruning is more sophisticated and the final goal of pruning. Under the assumption to prune 90% of the weights of our neural networks, we only run computational this rest of 10% weights. This would be hard pruning and in theory the hard pruned network would be much faster to compute than the original network with 100% of its weights.

At a very high level, the algorithm we need might look like this. We take a simple perceptron with 3 input neurons and its weights. Let's say w1 is to be pruned. A hard pruned algorithm requires a length for each output neuron in a perceptron, in this case length 3, and a new array to store the positions of each weight.
The hard pruning algorithm for the neural network swaps w1 with the last w3 in the position array and cut the length -1 so that the new length would be 2 and w1 drops out of the calculation completely. 

Sounds so simple, but it is not easy to understand all that to build this network.

![ggprune](https://raw.githubusercontent.com/grensen/ML_2020/main/figures/ggpruning.png)

The white lines (pruned weights) are not really used under the hood, it should just help to visualize what happened. Here is the hard pruning algorithm in action, the pruning technique which is used prunes after the half of the training steps 75% of the weights with magnitude pruning and trains the rest to the end. 
The test with pruning outperforms with a final accuracy of 93.62% my basic implementations with a test accuracy of 93.29%, hooray! All that with much more efficiency.

Another cool approach to magnitude pruning is described with [A foolproof way to shrink deep learning models](https://news.mit.edu/2020/foolproof-way-shrink-deep-learning-models-0430).
<p align="center">
  <img src="https://news.mit.edu/sites/default/files/styles/news_article__image_gallery/public/images/202004/learning_rate_rewinding.png" width="450" height="300" />
</p>

With [Movement Pruning: Adaptive Sparsity by Fine-Tuning(Paper Explained)](https://www.youtube.com/watch?v=nxEr4VNgYOE), the understanding of the importance of weights increases.
<p align="center">
  <img src="https://raw.githubusercontent.com/grensen/ML_2020/main/figures/movement_pruning_kilcher.png" width="524" height="294" />
</p>

Although pruning may seem exotic, it's a high-level technique for increasing understanding of what happens in neural networks. Each technique describes important properties to make the effects of weights more comprehensible.  

Common sense, lower weights have lower influence.

![ggstack](https://github.com/grensen/ML_2020/blob/main/figures/ggstacked.png)

Here are two twin networks which were separately trained and stacked after that. You can see the darker weights between these two network tunnels, they act as lightweight interconnections between them.

---

The most interesting work I did this year doesn't have a real name yet, but Leibniz was involved, it should be called Leibniz Network. Like Max Planck for quantum physics, Leibniz seems to me like the father of ML. 
His development of the binary system formed our digital age and with the differentiation calculus he opened the door for the existence of ML. [What are Differential Equations and how do they work?](https://www.youtube.com/watch?v=Em339AlejIs)
There are many other special people with these ideas that make our world possible, e.g., [David A. Huffman](https://en.wikipedia.org/wiki/Huffman_coding) to store our digital world, but all seem to be related to Leibniz.  

Did you know that Leibniz has a one in a zero on his tombstone? 
And did you know that the German Wikipedia says about Leibniz: "For Leibniz the 1 stood for God and the 0 for nothing"?

This is the Leibniz paradigm, the ReLU activation is the only activation function that makes this cut like this, it opens and closes dimensions depending on whether the value is positive. 
In the derivative multiplied by one, the world exists as before, but multiplied by zero, there is nothing. 

Let's say we have a neural network on our machine, it could be a smartphone or a supercomputer. We take all the available memory to create our neural network, which we call the ultimate network. The problem we can't solve even with the ReLU function is to compute the whole network in a realistic way. 
So we would need partial computations and many tricks and in the end a more fined tuned neural network could outperform this one.

On the other hand, we know that bigger models bring bigger improvements, we do it anyway. So the ultimate network would only compute the space we need, and the ReLU function is our gateway to let dimensions through or not. Time describes the size of the neural network.
Whew, I don't know what you think about this post, but this is a game changer for me.

---

A 2019 work can be called [custom connections](https://www.youtube.com/watch?v=ir6mgLMkezA&t=2s), which outperforms fully connected networks in my tests. 
The demo shows 1000 training steps for each network design. The first shows the fully connected approach we know. The second design connects the same weights in a random order, and the accuracy drops dramatically. The next design replaces only the last hidden to output layer with a fully connected layer to reduce the sharp drop in accuracy. 
Now the random connections go over two layers (purple from hidden1 to hidden3 and blue from hidden2 to output) and achieve the best accuracy yet.

<p align="center">
  <img src="https://raw.githubusercontent.com/grensen/ML_2020/main/figures/customconnect_ji.png" width="524" height="294" />
</p>

Fascinating things are parity patterns, the custom network in the figure takes 20% of the fully connected weights (green) and connects them in a parity pattern.
The accuracy goes from 50% to over 66%. The result has improved tremendously even though the same parameters were used in each example.
My plan is to go deeper with this work in 2021, if I can.  


One of the highlights of goodgame is the tailor-made pruning (right-click on a node in goodgame). The idea is to find an algorithm for each neural network that prunes a hidden neuron for efficiency reasons. 
Sounds silly, but it would help make neural networks more efficient and reduce power consumption.
The cool thing is that the algorithm uses the fastest way to sort the network after the pruning technique cuts out a node of a network with a simple shift. 
After the network comes out of this function, the network works as without this node, tailor-made.

What was surprising to me was the dependency of the positions of the hidden neurons, there was no dependency, not with a fully connected layer.
Let's go back to the ultimate network, if this network consists of only one hidden layer, the first neuron can be swapped with the last one. The result is the same.
You can try it yourself to dive into this dimension of understanding neural networks. Just take the code to start goodgame, create your model and train it. 
Then left-click on a hidden neuron. As a result, a copy of that neuron will come to the end of the layer. 
If you remove the source of your copy with a right click, the neuron is swapped and the result remains the same.

In our neural network universe, every node could be connected to every other node, the first could be connected to the last, 
and even more they could change their positions so that the first and the last node for this layer become real neighbors for our loops. It's relative.

---

I believe in energy, energy cannot normally be explained with logic, not for me.
You know the kind of energy when you deal with a woman or a dog in your life. 
There is this special energy, and to prevent a situation with common logic is an epic fail every time, especially with women. But with some kind of counter logic, the situation is simple, perhaps a form of non-linearity. 

My intention here is not to explain in words what I want to express, I want to express it with energy and I very much hope that this energy somehow reaches you, dear reader, for the 80s.

Starlink was that kind of energy to me. Unfortunately, I have not found an example that expresses what my loved ones and I saw in the sky, it was a breathtaking sight. 

<p align="center">
  <img src="https://www.basingstokegazette.co.uk/resources/images/11277965.png" width="200" height="200" />
</p>

Elon Musk was the reason for that, this guy plans Mars humans. Back to GPT-3, which was created by OpenAI. One founder of OpenAI was Elon Musk. 
Crazy story. But what if Mars humans and Earth humans starts war, and humans end up? I am ambivalent about many of these ideas. Werner Heisenberg showed that it can be good not to do something even if you can. 
On the other hand, we detect [5 out of 800 asteroids](https://youtu.be/4Wrc4fHSCpw) and could do nothing if threatened.

Elon Musk received the [Axel Springer award](https://youtu.be/AF2HXId2Xhg) in Berlin. He has truly earned it as one of our greatest visionaries. Hopefully, his decisions will be a bless for mankind.

#### This brings me to my personal Machine Learning Award 2020. ####
And the award goes to the person who put so many countless works into ML, there is no better one. James D. McCaffrey.
Many thanks to you for this wonderful year on your blog. I've never found a place that inspired me so much.    
<p align="center">
  <img src="https://raw.githubusercontent.com/grensen/ML_2020/main/figures/jamesmccaffrey.wordpress.com.png" />
</p>

Where others show scary things, you show the beauty.

A category on the blog is the Top Ten, here is mine:

---
[1. Another Look at Braess’s Paradox](https://jamesmccaffrey.wordpress.com/2019/11/06/another-look-at-braesss-paradox/) – How a smaller network run faster. The solution is the most interesting, in my opinion it's some kind of equation that shows the best way through a crisis.

---
[2. The Kermack-McKendrick SIR Disease Model Using C#](https://jamesmccaffrey.wordpress.com/2020/02/11/the-kermack-mckendrick-sir-disease-model-using-c/) – Most impressive is the date. This was the first SIR model after the situation became real. There were many important explanations, but this one was the most valuable to me.

---
[3. Implementing a Proportional Selection Function Using Roulette Wheel Selection](https://jamesmccaffrey.wordpress.com/2020/04/23/implementing-a-proportional-selection-function-using-roulette-wheel-selection/) – Remember the ultimate network and the problem that we cannot compute an entire layer at once. The roulette wheel selection technique could help find the best dimensions after a while.

---
[4. Deriving the Neural Weight Update Equation Using Common Sense Instead of Deep Math](https://jamesmccaffrey.wordpress.com/2020/03/01/deriving-the-neural-weight-update-equation-using-common-sense-instead-of-deep-math/) – Can you beat this explanation? An important tool for me is a simple calculator on my phone to understand what the functions were doing.

---
[5. The Difference Between the Norm of a Vector and the Distance Between Two Vectors](https://jamesmccaffrey.wordpress.com/2020/03/18/the-difference-between-the-norm-of-a-vector-and-the-distance-between-two-vectors/) – The norm of a vector can also be called L2 norm. A third form is the L1 norm, instead of the squares the absolute value is taken. Very interesting when using batch normalization.

---
[6. The Diffie–Hellman Key Exchange](https://jamesmccaffrey.wordpress.com/2020/07/29/the-diffie-hellman-key-exchange/) – Posts like these give me the confidence to dive deeper into a topic that was previously incomprehensible.

---
[7. Neural Network Resilient Back-Propagation (Rprop) using C#](https://jamesmccaffrey.wordpress.com/2015/03/20/neural-network-resilient-back-propagation-rprop-using-c/) – Almost forgotten technique, a good time for a reminder. 
Further techniques are: Kernel Logistic Regression, Radial Basis Function, The Perceptron, K-Means++, Gaussian Mixture Models, Naive Bayes, Genetic Algorithms, Evolutionary Optimization, MSO, PSO, RIO and so on... 

---
[8. Survivorship Bias](https://jamesmccaffrey.wordpress.com/2019/06/07/survivorship-bias/) – These effects, which affect our lives every day. The solution seems so simple, and then leads to undesirable effects.

---
[9. Spike Timing Dependent Plasticity Explained as Simply as Possible](https://jamesmccaffrey.wordpress.com/2019/12/27/spike-timing-dependent-plasticity-explained-as-simply-as-possible/) – The 3rd generation of neural networks, good to know what can come next. But it's a bit scary to dive in.

---
[10. PyTorch Has Too Many Ways To Do Anything: Ten Ways To Create The Same Tensor](https://jamesmccaffrey.wordpress.com/2020/06/05/pytorch-has-too-many-ways-to-do-anything-ten-ways-to-create-the-same-tensor/) – Python is not my language, but it is the most commonly used programming language in ML. Ten ways for one result seems like growing insanity to me. To get confidence with this mess, posts like this are gold.

---
Whoosh, that was a lot. Hope I was able to give you a good look at what I learned and developed about ML 2020. ML 2020 connects with the real world at increasing speed, it remains exciting. Many things remain a mystery to me and there is no end in sight, what a luck in the case of ML. [Belinda Carlisle says: Heaven Is A Place On Earth](https://youtu.be/7jrCfe3bYUY?list=RDMM7jrCfe3bYUY)

Leibniz thought we lived in the best of all possible worlds, and Musk says "Try harder", my tinnitus would agree!      

---
<p align="center">
  <img src="https://raw.githubusercontent.com/grensen/ML_2020/main/figures/111.png" />
</p>

*Imagine a building 1 meter wide and 24 meters high. Now imagine this building 18 meters wide and the same ratio. Middle: This is the Steinway Tower, also known as 111 w57st, the thinnest building in the world.
But how? Left: 192 high-density rock anchors screw the building into the rock. Right: Two gigantic shear walls make this iconic structure possible. With a few additional tricks, such as two differently calibrated, coordinated mass dampers, two open floors
and porosity at the top, this super slender tower is a new kind of skyscraper. [After this fascinating documentary, things could make sense](https://www.youtube.com/watch?v=WXY3HGThlvo&t=375s).*
