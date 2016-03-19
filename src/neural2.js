/**
 * Created by paularndt on 17/03/16.
 */
var RNN = {};

(function(globalAccess) {

    function assert(condition, message) {
        // from http://stackoverflow.com/questions/15313418/javascript-assert
        if (!condition) {
            message = message || "Assertion failed";
            if (typeof Error !== "undefined") {
                throw new Error(message);
            }
            throw message; // Fallback
        }
    }

    // random numbers utils
    var return_v = false;
    var v_val = 0.0;

    // helper function returns array of zeros of length n
    // and uses typed arrays if available
    var zeros = function(n) {
        if(typeof(n)==='undefined' || isNaN(n)) { return []; }
        if(typeof ArrayBuffer === 'undefined') {
            // lacking browser support
            var arr = new Array(n);
            for(var i=0;i<n;i++) { arr[i] = 0; }
            return arr;
        } else {
            return new Float64Array(n);
        }
    }

    var Matrix = function(rows, columns) {
        this.rows = rows;
        this.columns = columns;
        this.w = math.zeros(rows,columns);
        this.dw = math.zeros(rows,columns);
    }
    Matrix.prototype = {
        randomizeW: function(){
            this.w = math.random(math.size(this.w));
        }
    }

    // Transformer definitions
    var Graph = function(needs_backprop) {
        if(typeof needs_backprop === 'undefined') {needs_backprop = true;}
        this.needs_backprop = needs_backprop;

        // store a list of functions that perform backprop, in their forward pass order
        // so in backprop we will go backwards and evoke each one
        this.backprop = [];
    }
    Graph.prototype = {
        // backward execution of the functions in this.backprop
        backward: function() {
            for( var i = this.backprop.length-1; i >= 0; i--) {
               // this.backprop[i](); // one tick
            }
        },
        rowPluck: function(matrix, index) {
            // pluck a row of matrix with row-index index and return it as a column vector
            var columns = matrix.columns;
            var out = new Matrix(columns,1);
            for (var i = 0; i < columns; i++) {
                out.w.subset(math.index(i,0),matrix.w.subset(math.index(index,i)));
            }

            if(this.needs_backprop) {
                var backward = function() {
                    for (i = 0; i < columns; i++) {
                        matrix.dw.subset(math.index(index,i),math.add(out.dw.subset(math.index(i,0)),matrix.dw.subset(math.index(index,i))));
                    }
                }
                this.backprop.push(backward);
            }
            assert(out.w.size()[0] === out.rows && out.w.size()[1] === out.columns);
            return out;
        },
        tanh: function(matrix) {
            // tanh nonlinearity
            var out = new Matrix(matrix.rows, matrix.columns);
            out.w = matrix.w.map( function(value, index, matrix) {
                return math.tanh(value);
            });
            if (this.needs_backprop) {
                var backward = function() {
                    m.dw = out.dw.map(function (value, index, matrix) {
                        var mwi = out.w.subset(math.index(index));
                        return m.dw.subset(math.index(index)) + (1.0 - mwi * mwi) * value;
                    });
                }
                this.backprop.push(backward);
            }
            assert(out.w.size()[0] === out.rows && out.w.size()[1] === out.columns);
            return out;
        },
        relu: function(matrix) {
            var out = new Matrix(matrix.rows, matrix.columns);
            out.w = matrix.w.map( function(value, index, matrix) {
                return math.max(0, value); //relu
            });
            if (this.needs_backprop) {
                var backward = function() {
                    matrix.dw = out.dw.map(function (value, index, mat) {
                        return matrix.w.subset(math.index(index[0],index[1])) > 0 ? matrix.dw.subset(math.index(index[0],index[1])) + value : 0.0;
                    });
                }
                this.backprop.push(backward);
            }
            assert(out.w.size()[0] === out.rows && out.w.size()[1] === out.columns);
            return out;
        },
        mul: function(matrix1, matrix2) {
            assert(matrix1.columns === matrix2.rows, 'matrix multiplication dimensions misaligned');

            var out = new Matrix(matrix1.rows, matrix2.columns);
            out.w = math.multiply(matrix1.w, matrix2.w);

            if (this.needs_backprop) {
                var backward = function() {
                    for (var i = 0; i < matrix1.rows; i++) {
                        for (var j = 0; j < matrix2.columns; j++) {
                            var b = out.dw.subset(math.index(i,j));
                            for (var k = 0; k < matrix1.columns; k++) {
                                var mat1Value = math.add(matrix1.dw.subset(math.index(i,k)),math.multiply(b, matrix2.w.subset(math.index(k,j))));
                                matrix1.dw.subset(math.index(i,k), mat1Value);
                                var mat2Value = math.add(matrix2.dw.subset(math.index(k,j)),math.multiply(b, matrix1.w.subset(math.index(i,k))));
                                matrix2.dw.subset(math.index(k,j), mat2Value);
                            }
                        }
                    }
                }
                assert(out.w.size()[0] === out.rows && out.w.size()[1] === out.columns);
                this.backprop.push(backward);
            }
            return out;
        },
        add: function(matrix1, matrix2) {
            assert(matrix1.w.size === matrix2.w.size, 'matrix addition dimensions misaligned');
            var out = new Matrix(matrix1.rows, matrix1.columns);
            out.w = math.add(matrix1.w, matrix2.w);

            if (this.needs_backprop) {
                var backward = function() {
                    matrix1.dw = out.dw.map(function(value, index, matrix){
                        return value + matrix1.dw.subset(math.index(index[0],index[1]));
                    })
                    matrix2.dw = out.dw.map(function(value, index, matrix){
                        return value + matrix2.dw.subset(math.index(index[0], index[1]));
                    })
                }
                this.backprop.push(backward);
            }
            assert(out.w.size()[0] === out.rows && out.w.size()[1] === out.columns);
            return out;

        },
        eltmul: function(matrix1, matrix2) {
            assert(matrix1.w.size === matrix2.w.size, 'matrix multiplication dimensions misaligned');

            var out = new Matrix(matrix.rows, matrix.columns);
            out.w = math.multiply(matrix1.w, matrix2.w);

            if (this.needs_backprop) {
                var backward = function() {
                    matrix1.dw = out.dw.map(function(value, index, matrix){
                        return (value * matrix2.w.subset(math.index(index))) + matrix1.dw.subset(math.index(index));
                    })
                    matrix2.dw = out.dw.map(function(value, index, matrix){
                        return (value * matrix1.w.subset(math.index(index))) + matrix2.dw.subset(math.index(index));
                    })
                }
                this.backprop.push(backward);
            }
            assert(out.w.size()[0] === out.rows && out.w.size()[1] === out.columns);
            return out;
        },
    }


    var softmax = function(matrix) {
        var out = new Matrix(matrix.rows, matrix.columns); // probability volume
        var maxValue = -99999999;
        matrix.w.forEach(function(value, index, matrix) {
            if (value > maxValue) {maxValue = value;}
        });

        var s = 0.0;
        out.w = matrix.w.map(function(value, index, matrix) {
            var result = Math.exp(value - maxValue);
            s += result;
            return result;
        });
        out.w = out.w.map(function(value,index,matrix) {
            return value / s;
        });
        var count = 0;
        out.w.forEach(function(value, index, matrix) {
           count += value;
        });
        return out;

    }

    var Solver = function() {
        this.decay_rate = 0.999;
        this.smooth_eps = 1e-8;
        this.step_cache = {};
    }
    Solver.prototype = {
        step: function(model, stepSize, regc, clipValue) {
            // perform parameter update
            var solverStats = {};
            var numClipped = 0;
            var numTotalOperations = 0;
            var decay_rate = this.decay_rate;
            var smooth_eps = this.smooth_eps;
            for (var k in model) {
                if (model.hasOwnProperty(k)) {
                    var matrix = model[k]; // matrix reference
                    if(!(k in this.step_cache)) {this.step_cache[k] = new Matrix(matrix.rows, matrix.columns); }
                    var s = this.step_cache[k];
                    s.w.forEach(function(value,index,mat) {

                        var mathIndex = math.index(index[0],index[1]);

                        // rmsprop adaptive learning rate
                        var mdwi = matrix.w.subset(mathIndex);

                        var newSwi = s.w.subset(mathIndex) * decay_rate + (1.0 - decay_rate) * mdwi * mdwi;
                        s.w.subset(mathIndex, newSwi);

                        // gradient clip
                        if (mdwi > clipValue) {
                            mdwi = clipValue;
                            numClipped++;
                        }
                        if (mdwi < -clipValue) {
                            mdwi = -clipValue;
                            numClipped++;
                        }
                        numTotalOperations++;

                        //update (and regularize)
                        var newMwi = - stepSize * mdwi / math.sqrt(s.w.subset(mathIndex) + smooth_eps) - regc * matrix.w.subset(mathIndex);
                        matrix.w.subset(mathIndex, newMwi);
                        matrix.dw.subset(mathIndex, 0); // reset gradients for next iteration

                    })
                }
            }
            solverStats['ration_clipped'] = numClipped*1.0/numTotalOperations;
            return solverStats;
        }
    }

    var initRNN = function(input_size, hidden_sizes, output_size) {
        // hidden size should be a list

        var model= {};
        for (var d = 0; d < hidden_sizes.length; d++) {
            var prev_size = d === 0 ? input_size : hidden_sizes[d - 1];
            var hidden_size = hidden_sizes[d];
            model['Wxh'+d] = new Matrix(hidden_size, prev_size);
            model['Wxh'+d].randomizeW();
            model['Whh'+d] = new Matrix(hidden_size, hidden_size);
            model['Whh'+d].randomizeW();
            model['bhh'+d] = new Matrix(hidden_size, 1);
            model['bhh'+d].randomizeW();
        }
        // decoder params
        model['Whd'] = new Matrix(output_size, hidden_size);
        model['Whd'].randomizeW();
        model['bd'] = new Matrix(output_size,1);
        return model;
    }

    var forwardRNN = function(graph, model, hidden_sizes, x, prev) { // what is x?, better name for prev?

        // forward prop for a single tick of RNN
        // G is graph to append ops to
        // model contains RNN parameters
        // x is 1D column vector with observation
        // prev is a struct containing hidden activations from last step

        if (typeof prev.h === 'undefined') {
            var hidden_prevs = [];
            for (var i = 0; i < hidden_sizes.length; i++) {
                hidden_prevs.push(new Matrix(hidden_sizes[i], 1));
            }
        } else {
            var hidden_prevs = prev.h;
        }

        var hidden = [];
        for (var i = 0; i < hidden_sizes.length; i++) {
            var input_vector = i === 0 ? x : hidden[i-1];
            var hidden_prev = hidden_prevs[i];

            var h0 = graph.mul(model['Wxh'+i], input_vector);
            var h1 = graph.mul(model['Whh'+i], hidden_prev);
            var hidden_d = graph.relu(graph.add(graph.add(h0,h1), model['bhh'+i]));

            hidden.push(hidden_d);
        }

        // one decoder to outputs at end
        var output = graph.add(graph.mul(model['Whd'], hidden[hidden.length - 1]), model['bd']);

        // return cell memory, hidden representation and ouput
        return {'h': hidden, 'o': output};
    }

    var maxi = function(w) {
        // argmax of array w
        var maxv = w.subset(math.index(0,0));
        var maxix = [0,0];
        w.forEach(function(value, index, matrix) {
            if (value > maxv) {
                maxv = value;
                maxix = index;
            }
        })
        return maxix;
    }
    globalAccess.maxi = maxi;
    globalAccess.softmax = softmax;

    // classes
    globalAccess.Matrix = Matrix;

    globalAccess.initRNN = initRNN;
    globalAccess.forwardRNN = forwardRNN;

    // optimization
    globalAccess.Solver = Solver;
    globalAccess.Graph = Graph;


})(RNN);