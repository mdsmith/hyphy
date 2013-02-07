#include "parser.h"
#include "mathobj.h"

//SW: This calls the function with the opcode after it's been parsed
_PMathObj _MathObject::Execute (long opCode, _PMathObj p, _PMathObj p2, _PMathObj context)   // execute this operation with the second arg if necessary
{
    switch (opCode) {
    case HY_OP_CODE_NOT: // !
        return LNot();
    case HY_OP_CODE_NEQ: // !=
        if (ObjectClass () == HY_UNDEFINED) {
            if (p->ObjectClass () == HY_UNDEFINED)
                return new HY_CONSTANT_FALSE;
            else
                return new HY_CONSTANT_TRUE;
        }
        if (p->ObjectClass() == NUMBER)
            return NotEqual(p);
        return new HY_CONSTANT_TRUE;
    case HY_OP_CODE_IDIV: // $
        return longDiv(p);
    case HY_OP_CODE_MOD: // %
        return lDiv(p);
    case HY_OP_CODE_AND: // &&
        return LAnd(p);
    case HY_OP_CODE_MUL: // *
        if (p)
            return Mult(p);
        break;
    case HY_OP_CODE_ADD: // +
        if (p) 
            return Add(p);
        return Sum ();
    case HY_OP_CODE_SUB: // -
        if (p) 
            return Sub(p);
        return Minus();
        break;
    case HY_OP_CODE_DIV: // /
        return Div(p);
    case HY_OP_CODE_LESS: // <
        return Less(p);
    case HY_OP_CODE_LEQ: // <=
        return LessEq(p);
    case HY_OP_CODE_EQ: // ==
        {
            if (ObjectClass () == HY_UNDEFINED) {
                if (p->ObjectClass () == HY_UNDEFINED)
                    return new HY_CONSTANT_TRUE;
                else
                    return new HY_CONSTANT_FALSE;
            }
            if (p->ObjectClass() == NUMBER)
                return AreEqual(p);
            return new HY_CONSTANT_FALSE;
        }
        break;
    case HY_OP_CODE_GREATER: // >
        return Greater(p);
    case HY_OP_CODE_GEQ: // >=
        return GreaterEq(p);
    case HY_OP_CODE_ABS: // Abs
        return Abs();
    case HY_OP_CODE_ARCTAN: // Arctan
        return Arctan();
    case HY_OP_CODE_BETA: // Beta
        return Beta(p);
    case HY_OP_CODE_CCHI2: // CChi2
        return CChi2(p);
    case HY_OP_CODE_CGAMMADIST: // CGammaDist
        return CGammaDist(p,p2);
    case HY_OP_CODE_COS: // Cos
        return Cos();
    case HY_OP_CODE_COLUMNS: // Columns
    case HY_OP_CODE_ROWS: // Rows
        return new _Constant (0.0);
    case HY_OP_CODE_ERF: // Erf
        return Erf();
    case HY_OP_CODE_EXP: // Exp
        return Exp();
    case HY_OP_CODE_FORMAT: // Format
        return FormatNumberString(p,p2);
    case HY_OP_CODE_GAMMA: // Gamma
        return Gamma();
    case HY_OP_CODE_GAMMADIST: // GammaDist
        return GammaDist(p,p2);
    case HY_OP_CODE_IBETA: // IBeta
        return IBeta(p,p2);
    case HY_OP_CODE_IGAMMA: // IGamma
        return IGamma(p);
    case HY_OP_CODE_INVCHI2: // InvChi2
        return InvChi2(p);
    case HY_OP_CODE_LNGAMMA: // LnGamma
        return LnGamma();
    case HY_OP_CODE_LOG: // Log
        return Log();
    case HY_OP_CODE_MAX: // Max
        return Max(p);
    case HY_OP_CODE_MIN: // Min
        return Min(p);
    case HY_OP_CODE_RANDOM: // Random
        return Random(p);
    case HY_OP_CODE_SIMPLEX: // Simplex
        return Simplex();
    case HY_OP_CODE_SIN: // Sin
        return Sin();
    case HY_OP_CODE_SQRT: // Sqrt
        return Sqrt();
    case HY_OP_CODE_TAN: // Tan
        return Tan();
    case HY_OP_CODE_TIME: // Time
        return Time();
    case HY_OP_CODE_TYPE: // Type
        return Type();
    case HY_OP_CODE_ZCDF: // ZCDF
        return ZCDF();
    case HY_OP_CODE_POWER: // ^
        if (p)
            return Raise(p);
        break;
    case HY_OP_CODE_OR: // ||
        return LOr(p);
    }
    WarnNotDefined (this, opCode);
    return new _MathObject;
}

//__________________________________________________________________________________

BaseRef _MathObject::makeDynamic (void)
{
    return(_PMathObj)checkPointer(new _MathObject);
}

//SW: Why do we need a string for the type?
_PMathObj _MathObject::Type (void)
{
    _FString * ts = new _FString();
    switch (ObjectClass()) {

    case NUMBER:
        *(ts->theString)="Number";
        break;
    case MATRIX:
        *(ts->theString)="Matrix";
        break;
    case CONTAINER:
        *(ts->theString)="Container";
        break;
    case TREE_NODE:
        *(ts->theString)="TreeNode";
        break;
    case TREE:
        *(ts->theString)="Tree";
        break;
    case STRING:
        *(ts->theString)="String";
        break;
    case ASSOCIATIVE_LIST:
        *(ts->theString)="AssociativeList";
        break;
    case TOPOLOGY:
        *(ts->theString)="Topology";
        break;
    case POLYNOMIAL:
        *(ts->theString)="Polynomial";
        break;
    default:
        *(ts->theString) = "Unknown";

    }

    return ts;
}
