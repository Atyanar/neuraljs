var RNN = {};

(function(globalAccess) {
  "use strict";

    // UTILITIES
    //var math = require('mathjs');
    
    // math





    var RNNOutput = function() {
        var ys;
        var ps;

    }
    RNNOutput.prototype = {

    }

    var RNN = function(inputSize, hiddenSize, outputSize){

        // initializations

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        math.chain(3).add(4).multiply(2).done();

        // sequence length?
        // learning rate?
        // size of weight map and bias vector?

        // Matrices
        var Wxh; // size = hidden_size, vocab_size 
        var Whh; // size = hidden_size, hidden_size
        var Why; // size = vocab_size, hidden_size
        var bh; // hidden_size 
        var by; //vocab_size
    } 

  

    RNN.prototype = {
        computeBackwardPass: function(Cost) {
            var dWxh; //zeros_like Wxh
            var dWhh; // zeros_like Whh
            var dWhy; // zeros_like Why

        },
        computeForwardPass: function(inputMatrix) {
            // add t for time
            var xs; // initialize as zero-vector the size of the vocab
            //var xs[input] = 1; // set current char to one
            var hs; // softmax ( Wxh * xs + Whh * (hs[t-1] + bh))
            var ys; // Why * hs[t] + by
            var ps; //probabilities for next chars: exp(ys[t]) / sum(exp(ys[t]))


            var result = RNNOutput();
            return result;
        },
        lossFunction: function() {
            var xs; //input state
            var hs; // hidden state, set copy of initial hidden state
            var ys; // output state
            var ps; //

            var loss = 0;
        },

    }


    var newRNN = function() {
        var RNN = 20;
    }
    



    // functions accessible from outside
    globalAccess.RNN = RNN;
    globalAccess.Output = RNNOutput;

})(RNN);



// Batch Normalization? - faster learning, lower dependence from initial values...
// Activation function? USe ReLU
// Weight initialization? Use Xavier init
// use nesterov accelerated gradient to include momentum and avoid jittering 
// |-> or adagrap rmsprop (choose rmsprop!)
// |-> no, use Adam!!
// use inverted dropout?


