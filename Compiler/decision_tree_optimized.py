from Compiler.types import *
from Compiler.sorting import *
from Compiler.library import *
from Compiler.decision_tree import get_type, PrefixSum, PrefixSumR, PrefixSum_inv, PrefixSumR_inv, SortPerm, GroupSum, GroupPrefixSum, GroupFirstOne, output_decision_tree, pick, run_decision_tree, test_decision_tree
from Compiler import util, oram

from itertools import accumulate
import math

debug = False
debug_split = False
max_leaves = None

def GetSortPerm(keys, *to_sort, n_bits=None, time=False):
    """
        Compute and return secret shared permutation that stably sorts :param keys.
    """
    for k in keys:
        assert len(k) == len(keys[0])
    n_bits = n_bits or [None] * len(keys)
    bs = Matrix.create_from(sum([k.get_vector().bit_decompose(nb)
             for k, nb in reversed(list(zip(keys, n_bits)))], []))
    get_vec = lambda x: x[:] if isinstance(x, Array) else x
    res = Matrix.create_from(get_vec(x).v if isinstance(get_vec(x), sfix) else x
                             for x in to_sort)
    res = res.transpose()
    return radix_sort_from_matrix(bs, res)

def ApplyPermutation(perm, x):
    res = Array.create_from(x)
    reveal_sort(perm, res, False)
    return res

def ApplyInversePermutation(perm, x):
    res = Array.create_from(x)
    reveal_sort(perm, res, True)
    return res

def VectMax(key, *data, debug=False):
    def reducer(x, y):
        b = x[0]*y[1] > y[0]*x[1]
        return [b.if_else(xx, yy) for xx, yy in zip(x, y)]
    res = util.tree_reduce(reducer, zip(key, *data))
    return res

def Custom_GT_Fractions(x_num, x_den, y_num, y_den, n_threads=2):
    b = (x_num*y_den) > (x_den*y_num)
    b = Array.create_from(b).get_vector()
    return b

def GroupMax(g, keys, *x, debug=False):
    assert len(keys) == len(g)
    for xx in x:
        assert len(xx) == len(g)
    n = len(g)
    m = int(math.ceil(math.log(n, 2)))
    keys = Array.create_from(keys)
    x = [Array.create_from(xx) for xx in x]
    g_new = Array.create_from(g)
    g_old = g_new.same_shape()
    for d in range(m):
        w = 2 ** d
        g_old[:] = g_new[:]
        break_point()
        vsize = n - w
        g_new.assign_vector(g_old.get_vector(size=vsize).bit_or(
            g_old.get_vector(size=vsize, base=w)), base=w)
        b = Custom_GT_Fractions(keys.get_vector(size=vsize), x[0].get_vector(size=vsize), keys.get_vector(size=vsize, base=w), x[0].get_vector(size=vsize, base=w))
        for xx in [keys] + x:
            a = b.if_else(xx.get_vector(size=vsize),
                          xx.get_vector(size=vsize, base=w))
            xx.assign_vector(g_old.get_vector(size=vsize, base=w).if_else(
                xx.get_vector(size=vsize, base=w), a), base=w)
        break_point()
    t = sint.Array(len(g))
    t[-1] = 1
    t.assign_vector(g.get_vector(size=n - 1, base=1))
    return [GroupSum(g, t[:] * xx) for xx in [keys] + x]

def ComputeGini(g, x, y, notysum, ysum, debug=False):
    assert len(g) == len(y)
    y = [y.get_vector().bit_not(), y]
    u = [GroupPrefixSum(g, yy) for yy in y]
    s = [notysum, ysum]
    w = [ss - uu for ss, uu in zip(s, u)]
    us = sum(u)
    ws = sum(w)
    uqs = u[0] ** 2 + u[1] ** 2
    wqs = w[0] ** 2 + w[1] ** 2
    res_num = ws * uqs + us * wqs
    res_den = us * ws
    xx = x
    t = get_type(x).Array(len(x))
    t[-1] = MIN_VALUE
    t.assign_vector(xx.get_vector(size=len(x) - 1) + \
                    xx.get_vector(size=len(x) - 1, base=1))
    gg = g
    p = sint.Array(len(x))
    p[-1] = 1
    p.assign_vector(gg.get_vector(base=1, size=len(x) - 1).bit_or(
        xx.get_vector(size=len(x) - 1) == \
        xx.get_vector(size=len(x) - 1, base=1)))
    break_point()
    res_num = p[:].if_else(MIN_VALUE, res_num)
    res_den = p[:].if_else(1, res_den)
    t = p[:].if_else(MIN_VALUE, t[:])
    return res_num, res_den, t

MIN_VALUE = -10000

def FormatLayer(h, g, *a, debug=False):
    return CropLayer(h, *FormatLayer_without_crop(g, *a, debug=debug))

def FormatLayer_without_crop(g, *a, debug=False):
    for x in a:
        assert len(x) == len(g)
    v = [g.if_else(aa, 0) for aa in a]
    p = SortPerm(g.get_vector().bit_not())
    v = [p.apply(vv) for vv in v]
    return v

def CropLayer(k, *v):
    if max_leaves:
        n = min(2 ** k, max_leaves)
    else:
        n = 2 ** k
    return [vv[:min(n, len(vv))] for vv in v]

def TrainLeafNodes(h, g, y, NID, Label, debug=False):
    assert len(g) == len(y)
    assert len(g) == len(NID)
    return FormatLayer(h, g, NID, Label, debug=debug)

class TreeTrainer:
    def GetInversePermutation(self, perm, n_threads=2):
        res = Array.create_from(self.identity_permutation)
        reveal_sort(perm, res)
        return res

    def ApplyTests(self, x, AID, Threshold):
        m = len(x)
        n = len(AID)
        assert len(AID) == len(Threshold)
        for xx in x:
            assert len(xx) == len(AID)
        e = sint.Matrix(m, n)

        @for_range_multithread(self.n_threads, 1, m)
        def _(j):
            e[j][:] = AID[:] == j
        xx = sum(x[j]*e[j] for j in range(m)) 

        return 2 * xx.get_vector() < Threshold.get_vector()
    
    def TestSelection(self, g, x, y, pis, notysum, ysum, time=False):
        for xx in x:
            assert(len(xx) == len(g))
        assert len(g) == len(y)
        m = len(x)
        n = len(y)
        gg = g
        u, t = [get_type(x).Matrix(m, n) for i in range(2)]
        v = get_type(y).Matrix(m, n)
        s_num = get_type(y).Matrix(m, n)
        s_den = get_type(y).Matrix(m, n)
        a = sint.Array(n)

        notysum_arr = Array.create_from(notysum)
        ysum_arr = Array.create_from(ysum)

        @for_range_multithread(self.n_threads, 1, m)
        def _(j):
            single = not self.n_threads or self.n_threads == 1
            time = self.time and single
            if self.debug_selection:
                print_ln('run %s', j)
            u[j].assign_vector(x[j])
            v[j].assign_vector(y)
            reveal_sort(pis[j], u[j])
            reveal_sort(pis[j], v[j])
            s_num[j][:], s_den[j][:], t[j][:] = ComputeGini(g, u[j], v[j], notysum_arr, ysum_arr, debug=False)

        ss_num, ss_den, tt, aa = VectMax((s_num[j][:] for j in range(m)), (s_den[j][:] for j in range(m)), (t[j][:] for j in range(m)), range(m), debug=self.debug)

        aaa = get_type(y).Array(n)
        ttt = get_type(x).Array(n)

        GroupMax_num, GroupMax_den, GroupMax_ttt, GroupMax_aaa = GroupMax(g, ss_num, ss_den, tt, aa)

        f = sint.Array(n)
        f = (self.zeros.get_vector() == notysum).bit_or(self.zeros.get_vector() == ysum)
        aaa_vector, ttt_vector = f.if_else(0, GroupMax_aaa), f.if_else(MIN_VALUE, GroupMax_ttt)

        ttt.assign_vector(ttt_vector)
        aaa.assign_vector(aaa_vector)

        return aaa, ttt

    def SetupPerm(self, g, x, y):
        m = len(x)
        n = len(y)
        pis = get_type(y).Matrix(m, n)
        @for_range_multithread(self.n_threads, 1, m)
        def _(j):
            @if_e(self.attr_lengths[j])
            def _():
                pis[j][:] = self.GetInversePermutation(GetSortPerm([x[j]], x[j], y,
                                        n_bits=[1], time=time))
            @else_
            def _():
                pis[j][:] = self.GetInversePermutation(GetSortPerm([x[j]], x[j], y,
                                        n_bits=[None],
                                        time=time))
        return pis

    def UpdateState(self, g, x, y, pis, NID, b, k):
        m = len(x)
        n = len(y)
        q = SortPerm(b)
        
        y[:] = q.apply(y)
        NID[:] = 2 ** k * b + NID
        NID[:] = q.apply(NID)
        g[:] = GroupFirstOne(g, b.bit_not()) + GroupFirstOne(g, b)
        g[:] = q.apply(g)

        b_arith = sint.Array(n)
        b_arith = Array.create_from(b)
        
        @for_range_multithread(self.n_threads, 1, m)
        def _(j):
            x[j][:] = q.apply(x[j])
            b_permuted = ApplyPermutation(pis[j], b_arith)
            
            pis[j] = q.apply(pis[j])
            pis[j] = ApplyInversePermutation(pis[j], SortPerm(b_permuted).perm)

        return [g, x, y, NID, pis]

    @method_block
    def train_layer(self, k):
        g = self.g
        x = self.x
        y = self.y
        NID = self.NID
        pis = self.pis
        s0 = GroupSum(g, y.get_vector().bit_not())
        s1 = GroupSum(g, y.get_vector())
        a, t = self.TestSelection(g, x, y, pis, s0, s1)
        b = self.ApplyTests(x, a, t)
        p = SortPerm(g.get_vector().bit_not())
        self.nids[k], self.aids[k], self.thresholds[k]= FormatLayer_without_crop(g[:], NID, a, t, debug=self.debug)
        self.g, self.x, self.y, self.NID, self.pis = self.UpdateState(g, x, y, pis, NID, b, k)

        @if_(k >= (len(self.nids)-1))
        def _():
            self.label = Array.create_from(s0 < s1)

    def __init__(self, x, y, h, binary=False, attr_lengths=None,
                 n_threads=None):
        """ Securely Training Decision Trees Efficiently by `Bhardwaj et al.`_ : https://eprint.iacr.org/2024/1077.pdf

        This protocol has communication complexity O( mN logN + hmN + hN log N) which is an improvement of ~min(h, m, log N) over `Hamada et al.`_ : https://petsymposium.org/popets/2023/popets-2023-0021.pdf

        To run this protocol, at the root of the MP-SPDZ repo, run Scripts/compile-run.py -H HOSTS -E ring custom_data_dt $((2**13)) 11 4 -Z 3 -R 128

        :param x: Attribute values 
        :param y: Binary labels 
        :param h: Height of the decision tree
        :param binary: Binary attributes instead of continuous
        :param attr_lengths: Attribute description for mixed data
      (list of 0/1 for continuous/binary)
        :param n_threads: Number of threads 

        """
        assert not (binary and attr_lengths)
        if binary:
            attr_lengths = [1] * len(x)
        else:
            attr_lengths = attr_lengths or ([0] * len(x))
        for l in attr_lengths:
            assert l in (0, 1)
        self.attr_lengths = Array.create_from(regint(attr_lengths))
        Array.check_indices = False
        Matrix.disable_index_checks()
        for xx in x:
            assert len(xx) == len(y)
        m = len(x)
        n = len(y)
        self.g = sint.Array(n)
        self.g.assign_all(0)
        self.g[0] = 1
        self.NID = sint.Array(n)
        self.NID.assign_all(1)
        self.y = Array.create_from(y)
        self.x = Matrix.create_from(x)
        self.pis = sint.Matrix(m, n)
        self.nids, self.aids = [sint.Matrix(h, n) for i in range(2)]
        self.thresholds = self.x.value_type.Matrix(h, n)
        self.identity_permutation = sint.Array(n) 
        self.label = sintbit.Array(n)
        self.zeros = sint.Array(n)
        self.zeros.assign_all(0)
        self.n_threads = n_threads
        self.debug_selection = False
        self.debug_threading = True
        self.debug_gini = False
        self.debug_init = False
        self.debug_vectmax = False
        self.debug = False
        self.time = False

    def train(self):
        """ Train and return decision tree. """
        n = len(self.y)

        @for_range(n)
        def _(i):
            self.identity_permutation[i] = sint(i)

        h = len(self.nids)

        self.pis = self.SetupPerm(self.g, self.x, self.y)

        @for_range(h)
        def _(k):
            self.train_layer(k)
        return self.get_tree(h, self.label)
    
    def train_with_testing(self, *test_set, output=False):
        """ Train decision tree and test against test data.

        :param y: binary labels (list or sint vector)
        :param x: sample data (by attribute, list or
          :py:obj:`~Compiler.types.Matrix`)
        :param output: output tree after every level
        :returns: tree

        """
        for k in range(len(self.nids)):
            self.train_layer(k)
            tree = self.get_tree(k + 1, self.label)
            if output:
                output_decision_tree(tree)
            test_decision_tree('train', tree, self.y, self.x,
                               n_threads=self.n_threads)
            if test_set:
                test_decision_tree('test', tree, *test_set,
                                   n_threads=self.n_threads)
        return tree

    def get_tree(self, h, Label):
        Layer = [None] * (h + 1)
        for k in range(h):
            Layer[k] = CropLayer(k, self.nids[k], self.aids[k],
                                 self.thresholds[k])
        Layer[h] = TrainLeafNodes(h, self.g[:], self.y[:], self.NID, Label, debug=self.debug)
        return Layer

def DecisionTreeTraining(x, y, h, binary=False):
    return TreeTrainer(x, y, h, binary=binary).train()

class TreeClassifier:
    """ Tree classification that uses
    :py:class:`TreeTrainer` internally.

    :param max_depth: Depth of decision tree
    :param n_threads: Number of threads used

    """
    def __init__(self, max_depth, n_threads=None):
        self.max_depth = max_depth
        self.n_threads = n_threads

    @staticmethod
    def get_attr_lengths(attr_types):
        if attr_types == None:
            return None
        else:
            return [1 if x == 'b' else 0 for x in attr_types]

    def fit(self, X, y, attr_types=None):
        """ Train tree.

        :param X: Attribute values
        :param y: Binary labels

        """
        self.tree = TreeTrainer(
            X.transpose(), y, self.max_depth,
            attr_lengths=self.get_attr_lengths(attr_types),
            n_threads=self.n_threads).train()

    def output(self):
        output_decision_tree(self.tree)
    
    def fit_with_testing(self, X_train, y_train, X_test, y_test,
                         attr_types=None, output_trees=False, debug=False):
        """ Train tree with accuracy output after every level.

        :param X_train: training data with row-wise samples (sint/sfix matrix)
        :param y_train: training binary labels (sint list/array)
        :param X_test: testing data with row-wise samples (sint/sfix matrix)
        :param y_test: testing binary labels (sint list/array)
        :param attr_types: attributes types (list of 'b'/'c' for
          binary/continuous; default is all continuous)
        :param output_trees: output tree after every level
        :param debug: output debugging information

        """
        trainer = TreeTrainer(X_train.transpose(), y_train, self.max_depth,
                              attr_lengths=self.get_attr_lengths(attr_types),
                              n_threads=self.n_threads)
        trainer.debug = debug
        trainer.debug_gini = debug
        trainer.debug_threading = debug > 1
        self.tree = trainer.train_with_testing(y_test, X_test.transpose(),
                                               output=output_trees)

    def predict(self, X):
        """ Use tree for prediction.

        :param X: sample data with row-wise samples (sint/sfix matrix)
        :returns: sint array

        """
        res = sint.Array(len(X))
        @for_range(len(X))
        def _(i):
            res[i] = run_decision_tree(self.tree, X[i])
        return res
