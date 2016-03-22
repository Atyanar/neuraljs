var Neuraljs = {};

(function(globalAccess) {
    "use strict";

    // Utilities
    function assert(condition, message) {
        if (!condition) {
           message = message || 'Assertion failed';
            if (typeof Error !== 'undefined') {
                throw new Error(message);
            }
            throw message; // Fallback
        }
    }

    var gaussReturn = false;
    var gaussValueV = 0.0;

    var gaussRandom = function() {
        if (gaussReturn) {
            gaussReturn = false;
            return gaussValueV;
        }
        var u = 2*Math.random()-1;
        var v = 2*Math.random()-1;

        var r = u*u + v*v;

        if (r === 0 || r > 1) return gaussRandom();

        var c = Math.sqrt(-2*Math.log(r)/r);
        gaussValueV = v*c; // cached for return on next call
        gaussReturn = true;
        return u*c;
    };

    var zeros = function (n) {
        if (typeof(n) === 'undefined' || isNaN(n)) { return []; }
        if (typeof ArrayBuffer === 'undefined') {
            // no browser support
            var arr = [];
            for (var i = 0; i < n; i++) { arr[i] = 0; }
            return arr;
        }
        return new Float64Array(n);
    };

    var Matrix = function(rows, columns) {
        this.rows = rows;
        this.columns = columns;
        this.w = zeros(rows * columns);
        this.dw = zeros(rows * columns);
    };
    Matrix.prototype = {
        get: function(row, column) {
            var index = (this.rows * row) + column;
            assert((index >= 0) && (index < this.w.length));
            return this.w[index];
        },
        set: function(row, column, value) {
            var index = (this.rows * row) + column;
            assert((index >= 0) && (index < this.w.length));
            this.w[index] = value;
        },
        toJSON: function() {
            var json = {};
            json['rows'] = this.rows;
            json['columns'] = this.columns;
            json['w'] = this.w;
            return json;
        },
        fromJSON: function(json){
            this.rows = json.rows;
            this.columns = json.columns;
            this.w = zeros(this.rows * this.columns);
            this.dw = zeros(this.rows * this.columns);
            for (var i = 0, n = this.rows * this.columns; i < n; i++) {
                this.w[i] = json.w[i];
            }
        }
    };

    var RandomMatrix = function(rows, columns, std) {

        var fillRandom = function(matrix, low, high) {
            for(var i = 0, n = matrix.rows * matrix.columns; i < n; i++) {
                matrix.w[i] = Math.random()*(high-low)+low;
            }
        };

        var m = new Matrix(rows, columns);
        fillRandom(m, -std,std);
        return m;

    }

    /*
    Class responsible for mathematical operations on matrices. It automatically saves the
    necessary backpropagation functions, which will update the dw array of the matrix
     */
    var Graph = function(needsBackpropagation) {
        if (typeof needsBackpropagation === 'undefined') { needsBackpropagation = true; }
        this.needsBackpropagation = needsBackpropagation;

        // this stores a list of functions that perform the backpropagation in the right order
        this.backprop = [];
    }
    Graph.prototype = {
        performBackpropagation: function() {
            for ( var i = this.backprop.length - 1; i <= 0; i--) {
                this.backprop[i]; // execution of one function
            }
        },
        pluckRow: function(matrix, rowIndex) {
            // returns the row as a column vector
            assert((rowIndex >= 0) && (rowIndex <= matrix.rows));
            var columns = matrix.columns;
            var out = new Matrix(columns, 1);
            for (var i = 0; i < columns; i++) {
                out.w[i] = matrix.w[columns * rowIndex + i];
            }

            if (this.needsBackpropagation) {
                var backward = function() {
                    for (var i = 0; i < columns; i++) {
                        matrix.dw[columns * rowIndex + i] += out.dw[i];
                    }
                    this.backprop.push(backward);
                }
            }
            return out;
        },
        tanh: function(matrix) {
            var out = new Matrix(matrix.rows, matrix.columns);
            var n = matrix.w.length;
            for (var i = 0; i < n; i++) {
                out.w[i] = Math.tanh(matrix.w[i]);
            }

            if (this.needsBackpropagation) {
                var backward = function() {
                    var matrixWi = out.w[i];
                    matrix.dw[i] += (1.0 - matrixWi * matrixWi) * out.dw[i];
                }
            }
            this.backprop.push(backward);
        },
        sigmoid: function(matrix) {
            var out = new Matrix(matrix.rows, matrix.columns);
            var n = matrix.w.length;
            for (var i = 0; i < n; i++) {
                out.w[i] = 1.0/(1+Math.exp(-matrix.w[i]));
            }

            if (this.needsBackpropagation) {
                var backward = function() {
                    for (var i = 0; i < n; i++) {
                        var matrixWi = out.w[i];
                        m.dw[i] += mwi * (1.0 - mwi) * out.dw[i];
                    }
                }
                this.backprop.push(backward);
            }
            return out;
        },
        relu: function(matrix) {
            var out = new Matrix(matrix.rows, matrix.columns);
            var n = matrix.w.length;
            for (var i = 0; i < n; i++) {
                out.w[i] = Math.max(0, matrix.w[i]);
            }

            if (this.needsBackpropagation) {
                var backward = function() {
                    for (var i = 0; i < n; i++) {
                        matrix.dw[i] += matrix.w[i] > 0 ? out.dw[i] : 0.0;
                    }
                }
                this.backprop.push(backward);
            }
            return out;
        },
        mul: function(matrix1, matrix2) {
            assert(matrix1.columns === matrix2.rows, 'matrix dimensions not compatible for multiplication');

            var rows = matrix1.rows;
            var columns = matrix2.columns;
            var out = new Matrix(rows, columns);
            var dot;
            for (var i = 0; i < rows; i++) {
                for (var j = 0; j < columns; j++) {
                    dot = 0.0;
                    for (var k = 0; k < matrix1.columns; k++) {
                        dot += matrix1.w[matrix1.columns * i + k] * matrix2.w[matrix2.columns * k + j];
                    }
                    out.w[columns * i + j] = dot;
                }
            }

            if (this.needsBackpropagation) {
                var backward = function() {
                    var help;
                    for (i = 0; i < rows; i++) {
                        for (j = 0; j < columns; j++) {
                            help = out.dw[columns * i + j];
                            for (k = 0; k < matrix1.columns; k++) {
                                matrix1.dw[matrix1.columns * i + k] += matrix2.w[matrix2.columns * k + j] * help;
                                matrix2.dw[matrix2.columns * k + j] += matrix1.w[matrix1.columns * i + k] * help;
                            }
                        }
                    }
                }
                this.backprop.push(backward);
            }
            return out;
        },
        add: function(matrix1, matrix2) {
            assert(matrix1.w.length === matrix2.w.length && matrix1.rows === matrix2.rows, 'matrix dimensions not compatible for addition');

            var out = new Matrix(matrix1.rows, matrix1.columns);
            for (var i = 0; i < matrix1.w.length; i++) {
                out.w[i] = matrix1.w[i] + matrix2.w[i];
            }

            if (this.needsBackpropagation) {
                var backward = function() {
                    for (var i = 0; i < matrix1.w.length; i++) {
                        matrix1.dw[i] += matrix2.w[i] * out.dw[i];
                        matrix2.dw[i] += matrix1.w[i] * out.dw[i];
                    }
                }
                this.backprop.push(backward);
            }
            return out;
        },
        eltmul: function(matrix1, matrix2) {
            assert(matrix1.w.length === matrix2.w.length, 'matrix dimensions not compatible for element-multiplication');

            var out = new Matrix(matrix1.rows, matrix1.columns);
            for (var i = 0; i < matrix1.w.length; i++) {
                out.w[i] = matrix1.w[i] * matrix2.w[i];
            }

            if (this.needsBackpropagation) {
                var backward = function() {
                    for (var i = 0; i < matrix1.w.length; i++) {
                        matrix1.dw[i] += matrix2.w[i] * out.dw[i];
                        matrix2.dw[i] += matrix1.w[i] * out.dw[i];
                    }
                }
                this.backprop.push(backward);
            }
            return out;
        },
    }

    var softmax = function(matrix) {
        var out = new Matrix(matrix.rows, matrix.columns);
        var maxVal = -Infinity;
        for (var i = 0, n = matrix.w.length; i < n; i++) {
            if (matrix.w[i] > maxVal) maxVal = matrix.w[i];
        }

        var sum = 0.0;
        for (i = 0,n = matrix.w.length; i < n; i++) {
            out.w[i] = Math.exp(m.w[i] - maxVal);
            sum += out.w[i];
        }
        for (i = 0, n = matrix.w.length; i < n; i++) {
            out.w[i]  /= sum;
        }

        return out;
    }

    var NeuralNetwork = function(type) {
        if (typeof type === 'undefined' || type === 'LSTM') {

        }
        else if (type === 'RNN') {

        }
        else if (type === 'GRU') {

        }
        throw new Error('Unknown type for NeuralNetwork');
    }

})(Neuraljs);