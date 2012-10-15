/*

 HyPhy - Hypothesis Testing Using Phylogenies.

 Copyright (C) 1997-now
 Core Developers:
 Sergei L Kosakovsky Pond (spond@ucsd.edu)
 Art FY Poon    (apoon@cfenet.ubc.ca)
 Steven Weaver (sweaver@ucsd.edu)

 Module Developers:
 Lance Hepler (nlhepler@gmail.com)
 Martin Smith (martin.audacis@gmail.com)

 Significant contributions from:
 Spencer V Muse (muse@stat.ncsu.edu)
 Simon DW Frost (sdf22@cam.ac.uk)

 Permission is hereby granted, free of charge, to any person obtaining a
 copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 */

#ifdef MDSOCL

#include "calcnode.h"
#include <math.h>
#include "ocllikelib.h"
#include <iostream>
#include "helperFunctions.h"
#include "naiveFunctions.h"

void print_double_mat_ocl(double* m, int row_offset, int col_offset, int h, int w, int nr, int nc);

using namespace std;

double* temp_result;
double* my_temp_model;
double* temp_model_cache;

OCLlikeEval::OCLlikeEval(   long siteCount,
                            long alphabetDimension,
                            _Parameter* iNodeCache)
{
    init(siteCount, alphabetDimension, iNodeCache);
}

OCLlikeEval::OCLlikeEval()
{
    // Do nothing
}

void OCLlikeEval::init( long siteCount,
                        long alphabetDimension,
                        _Parameter* iNodeCache)
{
    this->siteCount = siteCount;
    this->alphabetDimension = alphabetDimension;
    this->iNodeCache = iNodeCache;
    gm = GPUMuller();
    contextSet = false;
    //cout << "OCLLikeEval initialized!" << endl;
}


void OCLlikeEval::setupContext()
{

    //cout << "flatParents: " << flatParents.lLength << endl;
    //cout << "flatNodes: " << flatNodes.lLength << endl;
    //cout << "flatCLeaves: " << flatCLeaves.lLength << endl;
    //cout << "flatLeaves: " << flatLeaves.lLength << endl;
    //cout << "flatTree: " << flatTree.lLength << endl;

    // XXX make this compatible with ambiguous nodes
    long treesize = alphabetDimension
                    * siteCount
                    * ( flatNodes.lLength
                        + flatLeaves.lLength);
/*
    treeCache = new double[ alphabetDimension
                            * siteCount
                            * ( flatNodes.lLength
                                + flatLeaves.lLength
                              )
                          ];
*/
    treeCache = new double[treesize];

    int leafThresh = alphabetDimension * siteCount * flatLeaves.lLength;
    for (int n = 0; n < flatLeaves.lLength; n++)
    {
        for (int s = 0; s < siteCount; s++)
        {
            for (int c = 0; c < alphabetDimension; c++)
            {
                treeCache[  n * siteCount * alphabetDimension
                            + s * alphabetDimension + c] = 0.0;
            }
        }
    }

    for (long n = 0; n < updateNodes.lLength; n++)
    {
        long nodeCode = updateNodes.lData[n];
        long parentCode = flatParents.lData[nodeCode];
        bool isLeaf = nodeCode < flatLeaves.lLength;
        if (isLeaf)
        {
            //cout << "NodeCode: " << nodeCode << endl;
            for (long s = 0; s < siteCount; s++)
            {
                long siteState = lNodeFlags[nodeCode*siteCount + s];
                if (siteState >= 0)
                {
                    treeCache[  nodeCode
                                * siteCount
                                * alphabetDimension
                                + s
                                * alphabetDimension
                                + siteState] = 1.0;
                                /*
                    cout    << "filling: "
                            <<  nodeCode
                                * siteCount
                                * alphabetDimension
                                + s
                                * alphabetDimension
                                + siteState
                            << endl;
                            */
                }
                else
                    cout << "no resolution?" << endl;
            }
        }
        // XXX ELSE resolve ambiguous nodes!
    }


    modelCache = new double[alphabetDimension
                            * alphabetDimension
                            * flatParents.lLength - 1];
    temp_model_cache = new double[  alphabetDimension
                                    * alphabetDimension
                                    * flatParents.lLength - 1];

    int ANR = (flatNodes.lLength + flatLeaves.lLength) * siteCount;
    int ANC = alphabetDimension;
    gm.set_A(treeCache, ANR, ANC);

    int BNR = (flatParents.lLength -1) * alphabetDimension;
    int BNC = alphabetDimension;
    gm.set_B(modelCache, BNR, BNC);

    gm.set_C(treeCache, ANR, ANC);
}

double OCLlikeEval::evaluate()
{
    for (int n =0; n < updateNodes.lLength; n++)
    {
        int nodeCode = updateNodes.lData[n];
        int uniqueNodeCode = nodeCode;
        int parentCode = flatParents.lData[n];
        bool isLeaf = nodeCode < flatLeaves.lLength;
        if (!isLeaf) nodeCode -= flatLeaves.lLength;
        //cout << "Updating nodeCode " << uniqueNodeCode << "'s model..." << endl;

        _Parameter* tMatrix = (isLeaf? ((_CalcNode*) flatCLeaves (nodeCode)):
                    ((_CalcNode*) flatTree
                    (nodeCode)))->GetCompExp(0)->theData;

        for (int pc = 0; pc < alphabetDimension; pc++)
        {
            for (int cc = 0; cc < alphabetDimension; cc++)
            {
                /*
                if (uniqueNodeCode == 9)
                {
                    cout << (double)(tMatrix[pc * alphabetDimension + cc]);
                }
                */
                //modelCache[uniqueNodeCode * alphabetDimension * alphabetDimension + pc *
                modelCache[ n
                            * alphabetDimension
                            * alphabetDimension
                            + pc
                            * alphabetDimension
                            + cc]
                            = (double)(tMatrix[ cc
                                                * alphabetDimension
                                                + pc]);
            }
            /*
            if (uniqueNodeCode == 9)
            {
                cout << endl;
            }
            */
        }
    }
    //cout << flatParents.lLength-1 << " total models" << endl;
    int updated_block = updateNodes.lLength * alphabetDimension;
    int BNR = (flatParents.lLength - 1) * alphabetDimension;
    int BNC = alphabetDimension;
    gm.update_B(modelCache, 0, 0, updated_block, BNC, BNR, BNC);

    int ARO, ACO, AH, UD, BW, BRO, BCO, CRO, CCO;

    for (int n = 0; n < updateNodes.lLength; n++)
    {

// XXX this is the correct method:
/*
        long    nodeCode   = updateNodes.lData [n],
                parentCode = flatParents.lData [nodeCode];
        cout    << "Starting in on the correct method for node "
                << n
                << " which is a branch from "
                << nodeCode
                << " to "
                << parentCode
                << " aka "
                << parentCode + flatLeaves.lLength
                << endl;
        // the INDEX of the parent node for the current node;
        // this list (a member of _TheTree is computed when the tree is created
        // for ((A,C)N1,D,(B,E)N2)Root this array will store
        // 0 (A), 0(C), 2(D), 1(B), 1(E), 2(N1), 2(N3), -1 (root)

        bool    isLeaf     = nodeCode < flatLeaves.lLength;

        int tagTemp = taggedInternals.lData[parentCode];

        //printf("Node: %li to %li\n", nodeCode, parentCode +
        //flatLeaves.lLength);

        if (!isLeaf) {
            nodeCode -=  flatLeaves.lLength;
        }

        _Parameter * parentConditionals = iNodeCache +  (parentCode  * siteCount) * alphabetDimension;
        // this is a convenience pointer into the global cache
        // each node will have siteCount*alphabetDimension contiguous doubles



        if (taggedInternals.lData[parentCode] == 0)
            // mark the parent for update and clear its conditionals if needed
        {
            taggedInternals.lData[parentCode]     = 1; // only do this once
            for (long k = 0, k3 = 0; k < siteCount; k++)
                for (long k2 = 0; k2 < alphabetDimension; k2++) {
                    parentConditionals [k3++] = 1.0;
                }
        }



        _Parameter  *       tMatrix = (isLeaf? ((_CalcNode*) flatCLeaves (nodeCode)):
                                       ((_CalcNode*) flatTree    (nodeCode)))->GetCompExp(0)->theData;
        // in the host code the transition matrix is retrieved from the _CalcNode object
        // in the devide code there will probably be another array to grab it from

        _Parameter  *       childVector; // the vector of conditional probabilities for
        // THIS node (nodeCode)

        if (!isLeaf) {
            childVector = iNodeCache + (nodeCode * siteCount) * alphabetDimension;
        }
        // if THIS node is internal, simply look up the conditional vector in the same cache
        // as the parent (just a different offset)

        //cout << "node: " << endl;
        cout << "do the math..." << endl;
        for (long siteID = 0; siteID < siteCount; siteID++,
                parentConditionals += alphabetDimension) {
            _Parameter  sum      = 0.0;

            if (isLeaf)
                // the leaves do NOT have a conditional probability vector because most of them are fully resolved
                // For example the codon AAG is going to map to (AAA)0,(AAC)0,(AAG)1,.....,(TTT)0, so we can
                // just store this as index 2, which says to put a '1' in the 2-nd index of the array (and assume the rest are 0)
                // this is stored in lNodeFlags
                // For ambiguous codons, e.g. AAR = {AAA,AAG}, the host code will resolve this to 1,0,1,0,...0 at prep time (because this will never change)
                // and stores it in the lNodeResolutions (say at index K) and puts -K-1 into lNodeFlags, so that we now have to look it up here
            {

                for (long p = 0; p < alphabetDimension; p++)
                    for (long c = 0; c < alphabetDimension; c++)
                        temp_model_cache[  n
                                        * alphabetDimension
                                        * alphabetDimension
                                        + p
                                        * alphabetDimension
                                        + c]
                                       = tMatrix[p*alphabetDimension +
                                       c];
                long siteState = lNodeFlags[nodeCode*siteCount + siteID] ;

                if (siteState >= 0)
                    // a single character state; sweep down the appropriate column
                    // note that one of the loops (over child state) drops out, since there is only one state
                {
                    long matrixIndex =  siteState;
                    for (long k = 0; k < alphabetDimension; k++, matrixIndex += alphabetDimension) {
                        parentConditionals[k] *= tMatrix[matrixIndex];
                    }
                    continue; // nothing else to do, move to the next site
                } else {
                    childVector = lNodeResolutions->theData + (-siteState-1) * alphabetDimension;
                }
                // look up the resolution for the ambugious node -- this will have to be on the device as well
                // but can be in constant memory
            }

            _Parameter *matrixPointer = tMatrix;

            for (long p = 0; p < alphabetDimension; p++) {
                _Parameter      accumulator = 0.0;

                for (long c = 0; c < alphabetDimension; c++) {
                    accumulator +=  matrixPointer[c]   * childVector[c];
                    temp_model_cache[  n
                                        * alphabetDimension
                                        * alphabetDimension
                                        + p
                                        * alphabetDimension
                                        + c]
                                       = matrixPointer[c];
                }

                matrixPointer                 += alphabetDimension;

                sum += (parentConditionals[p] *= accumulator);
                // XXX temp printer
                //cout << parentConditionals[p] << ", ";
            }
            //cout << endl;

            // if (sum < small_number) -- handle underflow

            childVector    += alphabetDimension;
            // shift the position in childvector to the next site
        }

        if (!isLeaf)
            childVector = iNodeCache + (nodeCode * siteCount) * alphabetDimension;




// XXX before this comment is the correct method, after is my method
// XXX fix tagged internals before uncommenting the above
        cout << "Starting in on my method: " << endl;
        taggedInternals.lData[parentCode] = tagTemp;
*/


        long nodeCode = updateNodes.lData[n];
        long parentCode = flatParents.lData[nodeCode];
        nodeCode = updateNodes.lData[n];

        //cout << "flatLeaves.lLength: " << flatLeaves.lLength;
        //cout << "nodeID: " << n << endl;
        //cout << "nodeCode: " << nodeCode << endl;
        //cout << "parentCode: " << parentCode << endl;

        // At this point you would normally subtract the number of leaves
        // from nodeCode, but as we are recreating the whole tree it is fine
        // if we leave nodeCode whole
        ARO = nodeCode * siteCount;
        ACO = 0;
        AH = siteCount;
        UD = alphabetDimension;


/*
        cout    << "ARO, ACO, AH, UD: "
                << ARO
                << ", "
                << ACO
                << ", "
                << AH
                << ", "
                << UD
                << endl;
*/
        gm.bound_A(ARO, ACO, AH, UD);


/*
        if (nodeCode >= flatLeaves.lLength)
        {
            int nc_temp = nodeCode - flatLeaves.lLength;
            double* childVector = iNodeCache + (nc_temp * siteCount) * alphabetDimension;
            double* A_temp = gm.get_A();
            bool same = true;
            for (int i = 0; i < siteCount * alphabetDimension; i++)
            {
                if (childVector[i] != A_temp[i]) same = false;
            }
            if (same)
                cout << "Yay, they are the same" << endl;
            else cout << "Boo, they are different" << endl;
            exit(1);
        }
*/

        BRO = n * alphabetDimension;
        BCO = 0;
        BW = alphabetDimension;

/*
        cout    << "BRO, BCO, UD, BW: "
                << BRO
                << ", "
                << BCO
                << ", "
                << UD
                << ", "
                << BW
                << endl;
*/
        gm.bound_B(BRO, BCO, UD, BW);


        // XXX compare models:
        /*
        cout << "Comparing models..." << endl;
        double* temp_B = gm.get_double_bound_B();
        for (int c = 0; c < alphabetDimension*alphabetDimension; c++)
        {
            if (fabs(temp_B[c] - tMatrix[c])/non_zero_max(temp_B[c],
            tMatrix[c]) > 1e-6)
            {
                cout << "Model busted dude!" << endl;
                cout << "at c= " << c << endl;
                for (   int pc = 0;
                        pc < alphabetDimension*alphabetDimension;
                        pc++)
                {
                    cout << temp_B[pc] << "v" << tMatrix[pc] << ",";
                    if (pc % alphabetDimension == alphabetDimension-1)
                        cout << endl;
                    if (pc == c)
                        cout << "HERE";
                }
                exit(1);
            }
        }
        // XXX compare A's
        double* temp_A;
        if (!isLeaf)
        {
            cout << "Comparing non-leaf Children..." << endl;
            temp_A = gm.get_double_bound_A();
            for (int c = 0; c < siteCount*alphabetDimension; c++)
            {
                if (temp_A[c] != childVector[c])
                {
                    cout << "child busted dude!" << endl;
                    cout << "at c= " << c << endl;
                    for (   int pc = 0;
                            pc < siteCount*alphabetDimension;
                            pc++)
                    {
                        cout << temp_A[pc] << "v" << childVector[pc] << ",";
                        if (pc % alphabetDimension == alphabetDimension-1)
                            cout << endl;
                        if (pc == c)
                            cout << "HERE";
                    }
                    exit(1);
                }
            }
        }
        else
        {
            cout << "Comparing leaf Children..." << endl;
            temp_A = gm.get_double_bound_A();
            for (int c = 0; c < siteCount; c++)
            {
                long siteState = lNodeFlags[nodeCode*siteCount + c] ;
                if (temp_A[c*alphabetDimension + siteState] != 1)
                {
                    cout << "child busted dude!" << endl;
                    cout << "at c= " << c << endl;
                    cout << "sitestate = " << siteState << endl;
                    for (   int pc = 0;
                            pc < siteCount*alphabetDimension;
                            pc++)
                    {
                        siteState = lNodeFlags[nodeCode*siteCount +
                        (pc/siteCount)];
                        cout << temp_A[pc] << "v";
                        if (pc % alphabetDimension == siteState)
                            cout << "1";
                        else
                            cout << "0";
                        cout << ",";
                        if (pc % alphabetDimension == alphabetDimension-1)
                            cout << endl;
                        if (pc == c)
                            cout << "HERE";
                    }
                    exit(1);
                }
            }
        }
        */


        // However just as we would normally subtract the number of leaves
        // from nodeCode but we don't, we now have to add the number of
        // leaves to parentCode so that stuff is saved in the correct place
        CRO = (parentCode + flatLeaves.lLength) * siteCount;
        CCO = 0;
        //cout << "Node: " << nodeCode << " to: " << parentCode +
        //flatLeaves.lLength << endl;

/*
        cout    << "CRO, CCO: "
                << CRO
                << ", "
                << CCO
                << endl;
*/

        if (taggedInternals.lData[parentCode] == 0)
        {
            //cout << "marking done..." << endl;
            taggedInternals.lData[parentCode] = 1;
            gm.set_overwrite(true);
            //cout << "overwrite!" << endl;
        }
        else
        {
            gm.set_overwrite(false);
            //cout << "don't overwrite!" << endl;
        }

        gm.eval_C(CRO, CCO, AH, BW);

        // XXX compare C's
        /*
        double* temp_C = gm.get_double_bound_C();
        parentConditionals = iNodeCache +  (parentCode  * siteCount) * alphabetDimension;
        cout << "Comparing parents..." << endl;
        for (int c = 0; c < siteCount*alphabetDimension; c++)
        {
            if (temp_C[c] != parentConditionals[c])
            {
                cout << "result is busted, dude!" << endl;
                cout << "Child: " << endl;
                print_double_mat_ocl(   temp_A,
                                        0,
                                        0,
                                        siteCount,
                                        alphabetDimension,
                                        siteCount,
                                        alphabetDimension);
                cout << "Model:" << endl;
                print_double_mat_ocl(   temp_B,
                                        0,
                                        0,
                                        alphabetDimension,
                                        alphabetDimension,
                                        alphabetDimension,
                                        alphabetDimension);
                cout << "tMatrix:" << endl;
                print_double_mat_ocl(   tMatrix,
                                        0,
                                        0,
                                        alphabetDimension,
                                        alphabetDimension,
                                        alphabetDimension,
                                        alphabetDimension);
                cout << "My Result:" << endl;
                print_double_mat_ocl(   temp_C,
                                        0,
                                        0,
                                        siteCount,
                                        alphabetDimension,
                                        siteCount,
                                        alphabetDimension);
                cout << "Correct Result:" << endl;
                print_double_mat_ocl(   parentConditionals,
                                        0,
                                        0,
                                        siteCount,
                                        alphabetDimension,
                                        siteCount,
                                        alphabetDimension);
                cout << "My Naive Result:" << endl;
                double* naive_result = naive_matrix_multiply_double(
                                        temp_A,
                                        temp_B,
                                        0,
                                        0,
                                        0,
                                        0,
                                        siteCount,
                                        siteCount,
                                        alphabetDimension,
                                        alphabetDimension,
                                        alphabetDimension,
                                        alphabetDimension);
                print_double_mat_ocl(   naive_result,
                                        0,
                                        0,
                                        siteCount,
                                        alphabetDimension,
                                        siteCount,
                                        alphabetDimension);
                cout << "Correct Naive Result:" << endl;
                double* correct_result = naive_matrix_multiply_double(
                                        temp_A,
                                        tMatrix,
                                        0,
                                        0,
                                        0,
                                        0,
                                        siteCount,
                                        siteCount,
                                        alphabetDimension,
                                        alphabetDimension,
                                        alphabetDimension,
                                        alphabetDimension);
                print_double_mat_ocl(   correct_result,
                                        0,
                                        0,
                                        siteCount,
                                        alphabetDimension,
                                        siteCount,
                                        alphabetDimension);
                cout << "Compare at c= " << c << endl;
                for (   int pc = 0;
                        pc < siteCount*alphabetDimension;
                        pc++)
                {
                    cout << temp_C[pc] << "v" << parentConditionals[pc] << ",";
                    if (pc % alphabetDimension == alphabetDimension-1)
                        cout << endl;
                    if (pc == c)
                        cout << "HERE";
                }
                exit(1);
            }
        }
        cout << "done with this node..." << endl;
        */

/*
        double * parentConditionals = iNodeCache +  (parentCode  * siteCount) * alphabetDimension;
        print_double_mat_ocl(   parentConditionals,
                                0,
                                0,
                                siteCount,
                                alphabetDimension,
                                siteCount,
                                alphabetDimension);
                                */

    }

    int rootRO = (flatLeaves.lLength + flatNodes.lLength - 1) * siteCount;
    int rootCO = 0;
    int rootH = siteCount;
    int rootW = alphabetDimension;
    // XXX check to make sure there is an option for slicing without mangling
    // memory
/*
    cout    << "rootRO, rootCO: "
            << rootRO
            << ", "
            << rootCO
            << ", "
            << "rootH, rootW: "
            << rootH
            << ", "
            << rootW
            << endl;
*/
    double* result = gm.get_C_double(rootRO, rootCO, rootH, rootW);
    /*
    int ANR = (flatNodes.lLength + flatLeaves.lLength) * siteCount;
    int ANC = alphabetDimension;
    gm.set_A(treeCache, ANR, ANC);
    gm.set_C(treeCache, ANR, ANC);
    */
    /*
    temp_result = gm.get_C_double(
                                    //flatLeaves.lLength * siteCount,
                                    0,
                                    0,
                                    flatNodes.lLength*siteCount,
                                    alphabetDimension);
                                    */
    return process_results(result);
}

double OCLlikeEval::process_results(double* root_conditionals)
{
    //cout << "Root Conditionals in processor: " << endl;
    double result = 0.0;
    long root_index = 0;
    for (long s = 0; s < siteCount; s++)
    {
        double accumulator = 0.0;
        for (long pc = 0; pc < alphabetDimension; pc++)
        {
            //cout << root_conditionals[root_index] << " ";
            accumulator += root_conditionals[root_index] * theProbs[pc];
            root_index++;
        }
        //cout << endl;
        result += log(accumulator) * theFrequencies[s];
    }
    /*
    cout << "Result: " << result << endl;
    if (result > -3433.66)
        exit(1);
        */
    return result;
}

double OCLlikeEval::eval_likelihood(    _SimpleList& updateNodes,
                                        _SimpleList& flatParents,
                                        _SimpleList& flatNodes,
                                        _SimpleList& flatCLeaves,
                                        _SimpleList& flatLeaves,
                                        _SimpleList& flatTree,
                                        _Parameter* theProbs,
                                        _SimpleList& theFrequencies,
                                        long* lNodeFlags,
                                        _SimpleList& taggedInternals,
                                        _GrowingVector* lNodeResolutions)
{
    this->updateNodes = updateNodes;
    this->taggedInternals = taggedInternals;
    this->theFrequencies = theFrequencies;
    if (!contextSet)
    {
        this->theProbs = theProbs;
        this->flatNodes = flatNodes;
        this->flatCLeaves = flatCLeaves;
        this->flatLeaves = flatLeaves;
        this->flatTree = flatTree;
        this->flatParents = flatParents;
        this->lNodeFlags = lNodeFlags;
        this->lNodeResolutions = lNodeResolutions;
        setupContext();
        contextSet = true;
    }

    //return evaluate();
    /*
    cout << "Before: " << endl;
    for (int i = 0; i < taggedInternals.lLength; i++)
    {
         cout << this->taggedInternals.lData[i] << endl;
    }
    */

/*
    double correct_answer = naive_likelihood();
    //cout << "After: " << endl;
    for (int i = 0; i < taggedInternals.lLength; i++)
    {
         //cout << this->taggedInternals.lData[i] << endl;
         this->taggedInternals.lData[i] = 0;
    }
*/
    double my_answer = evaluate();

/*
    // iNodeCache vs temp_result
    bool pass = true;
    for (   long int i =    flatLeaves.lLength
                            * alphabetDimension
                            * alphabetDimension;
            i < (flatNodes.lLength -1) * alphabetDimension * alphabetDimension;
            i++)
    {
        if (modelCache[i] != temp_model_cache[i])
        {
            pass = false;
            long int node = i
                            / ( alphabetDimension
                                * alphabetDimension);
            long int sub_i = i % node;
            long int pc = sub_i / alphabetDimension;
            long int cc = sub_i % alphabetDimension;
            cout << "Something isn't the same in the model!:" << endl;
            cout    << "i: "
                    << i
                    << endl;
            cout    << "correct: "
                    << modelCache[i]
                    << " mine: "
                    << temp_model_cache[i]
                    << endl;
            exit(1);
        }
    }
    pass = true;
    */
    /*
    cout << "correct_model:" << endl;
    print_double_mat_ocl(   temp_model_cache,
                            //flatLeaves.lLength * alphabetDimension,
                            0,
                            0,
                            (flatParents.lLength-1)*alphabetDimension,
                            alphabetDimension,
                            (flatParents.lLength-1) * alphabetDimension,
                            alphabetDimension);
    cout << "correct_result:" << endl;
    print_double_mat_ocl(   iNodeCache,
                            0,
                            0,
                            flatNodes.lLength * siteCount,
                            alphabetDimension,
                            flatNodes.lLength * siteCount,
                            alphabetDimension);
    */
    /*
    cout << "temp_result:" << endl;
    print_double_mat_ocl(   temp_result,
                            0,
                            0,
                            siteCount,
                            alphabetDimension,
                            flatNodes.lLength * siteCount,
                            alphabetDimension);
    */
    /*
    for (   long int i = 0;
            i < flatNodes.lLength * siteCount * alphabetDimension;
            i++)
    {
        if (iNodeCache[i] != temp_result[i])
        {
            pass = false;
            long int total_site = i / alphabetDimension;
            long int node = total_site / siteCount;
            long int site = total_site % siteCount;
            long int letter = i % alphabetDimension;
            cout << "Something isn't the same in the nodecache!:" << endl;
            printf( "i: %li node: %li site: %li letter: %li\n",
                    i,
                    node,
                    site,
                    letter);
            cout    << "correct: "
                    << iNodeCache[i]
                    << " mine: "
                    << temp_result[i]
                    << endl;
            cout << "correct_result:" << endl;
    */
            /*
            print_double_mat_ocl(   iNodeCache,
                                    0,
                                    0,
                                    siteCount,
                                    alphabetDimension,
                                    flatNodes.lLength * siteCount,
                                    alphabetDimension);
            cout << "temp_result:" << endl;
            print_double_mat_ocl(   temp_result,
                                    0,
                                    0,
                                    siteCount,
                                    alphabetDimension,
                                    flatNodes.lLength * siteCount,
                                    alphabetDimension);
            exit(1);
        }
    }
            */


    //cout << "My answer: " << my_answer << " correct answer: " <<
    //correct_answer << endl;
    //return correct_answer;
    return my_answer;
}


double OCLlikeEval::naive_likelihood()
{
    for  (long nodeID = 0; nodeID < updateNodes.lLength; nodeID++) {
        long    nodeCode   = updateNodes.lData [nodeID],
                parentCode = flatParents.lData [nodeCode];
        // the INDEX of the parent node for the current node;
        // this list (a member of _TheTree is computed when the tree is created
        // for ((A,C)N1,D,(B,E)N2)Root this array will store
        // 0 (A), 0(C), 2(D), 1(B), 1(E), 2(N1), 2(N3), -1 (root)

        bool    isLeaf     = nodeCode < flatLeaves.lLength;

        //printf("Node: %li to %li\n", nodeCode, parentCode +
        //flatLeaves.lLength);

        if (!isLeaf) {
            nodeCode -=  flatLeaves.lLength;
        }

        _Parameter * parentConditionals = iNodeCache +  (parentCode  * siteCount) * alphabetDimension;
        // this is a convenience pointer into the global cache
        // each node will have siteCount*alphabetDimension contiguous doubles



        if (taggedInternals.lData[parentCode] == 0)
            // mark the parent for update and clear its conditionals if needed
        {
            taggedInternals.lData[parentCode]     = 1; // only do this once
            for (long k = 0, k3 = 0; k < siteCount; k++)
                for (long k2 = 0; k2 < alphabetDimension; k2++) {
                    parentConditionals [k3++] = 1.0;
                }
        }



        _Parameter  *       tMatrix = (isLeaf? ((_CalcNode*) flatCLeaves (nodeCode)):
                                       ((_CalcNode*) flatTree    (nodeCode)))->GetCompExp(0)->theData;
        // in the host code the transition matrix is retrieved from the _CalcNode object
        // in the devide code there will probably be another array to grab it from

        _Parameter  *       childVector; // the vector of conditional probabilities for
        // THIS node (nodeCode)

        if (!isLeaf) {
            childVector = iNodeCache + (nodeCode * siteCount) * alphabetDimension;
        }
        // if THIS node is internal, simply look up the conditional vector in the same cache
        // as the parent (just a different offset)

        //cout << "node: " << endl;
        for (long siteID = 0; siteID < siteCount; siteID++,
                parentConditionals += alphabetDimension) {
            _Parameter  sum      = 0.0;

            if (isLeaf)
                // the leaves do NOT have a conditional probability vector because most of them are fully resolved
                // For example the codon AAG is going to map to (AAA)0,(AAC)0,(AAG)1,.....,(TTT)0, so we can
                // just store this as index 2, which says to put a '1' in the 2-nd index of the array (and assume the rest are 0)
                // this is stored in lNodeFlags
                // For ambiguous codons, e.g. AAR = {AAA,AAG}, the host code will resolve this to 1,0,1,0,...0 at prep time (because this will never change)
                // and stores it in the lNodeResolutions (say at index K) and puts -K-1 into lNodeFlags, so that we now have to look it up here
            {

                for (long p = 0; p < alphabetDimension; p++)
                    for (long c = 0; c < alphabetDimension; c++)
                        temp_model_cache[  nodeID
                                        * alphabetDimension
                                        * alphabetDimension
                                        + p
                                        * alphabetDimension
                                        + c]
                                       = tMatrix[p*alphabetDimension +
                                       c];
                long siteState = lNodeFlags[nodeCode*siteCount + siteID] ;

                if (siteState >= 0)
                    // a single character state; sweep down the appropriate column
                    // note that one of the loops (over child state) drops out, since there is only one state
                {
                    long matrixIndex =  siteState;
                    for (long k = 0; k < alphabetDimension; k++, matrixIndex += alphabetDimension) {
                        parentConditionals[k] *= tMatrix[matrixIndex];
                    }
                    continue; // nothing else to do, move to the next site
                } else {
                    childVector = lNodeResolutions->theData + (-siteState-1) * alphabetDimension;
                }
                // look up the resolution for the ambugious node -- this will have to be on the device as well
                // but can be in constant memory
            }

            _Parameter *matrixPointer = tMatrix;

            for (long p = 0; p < alphabetDimension; p++) {
                _Parameter      accumulator = 0.0;

                for (long c = 0; c < alphabetDimension; c++) {
                    accumulator +=  matrixPointer[c]   * childVector[c];
                    temp_model_cache[  nodeID
                                        * alphabetDimension
                                        * alphabetDimension
                                        + p
                                        * alphabetDimension
                                        + c]
                                       = matrixPointer[c];
                }

                matrixPointer                 += alphabetDimension;

                sum += (parentConditionals[p] *= accumulator);
                // XXX temp printer
                //cout << parentConditionals[p] << ", ";
            }
            //cout << endl;

            // if (sum < small_number) -- handle underflow

            childVector    += alphabetDimension;
            // shift the position in childvector to the next site
        }
    }

    // now just process the root and return the likelihood

    _Parameter  * rootConditionals = iNodeCache + alphabetDimension * ((flatTree.lLength-1)  * siteCount),
                  // the root is always the LAST internal node in all lists
                  result = 0.0;


    for (long siteID = 0; siteID < siteCount; siteID++) {
        _Parameter accumulator = 0.;
        for (long p = 0; p < alphabetDimension; p++,rootConditionals++) {
            accumulator += *rootConditionals * theProbs[p];
        }
        // theProbs is a member variable of the tree, which basically determines
           //what probability there is to observe a given character at the root
           //in simple cases it is fixed for the duration of optimization, but for more
           //complex models it may change from iteration to iteration

        result += log(accumulator) * theFrequencies [siteID];
        // correct for the fact that identical alignment columns may appear more than once
    }

    return result;


}


OCLlikeEval::~OCLlikeEval()
{
    int i = 0;
}

void print_double_mat_ocl(double* m, int row_offset, int col_offset, int h, int w, int nr, int nc)
{
    cout << "{";
    for (int r = row_offset; r < row_offset + h; r++)
    {
        cout << "{";
        for (int c = col_offset; c < col_offset + w; c++)
        {
            if (c > col_offset && c < col_offset + w)
                cout << ",";
            printf("%g", m[r*nc + c]);
        }
        if (r < row_offset + h-1)
#if defined WOLFRAM
            cout << "},";
#else
            cout << "}," << endl;
#endif
        else
            cout << "}";
    }
    cout << "}" << endl;
}
/*
*/


#endif
