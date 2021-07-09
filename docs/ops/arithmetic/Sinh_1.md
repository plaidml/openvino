## Sinh <a name="Sinh"></a> {#openvino_docs_ops_arithmetic_Sinh_1}

**Versioned name**: *Sinh-1*

**Category**: Arithmetic unary operation

**Short description**: *Sinh* performs element-wise hyperbolic sine (sinh) operation on a given input tensor

**Detailed description**: *Sinh* performs element-wise hyperbolic sine (sinh) operation on a given input tensor, based on the following mathematical formula:

\f[
a_{i} = sinh(a_{i})
\f]

**Attributes**: *Sinh* operation has no attributes.

**Inputs**

* **1**: An tensor of type *T*. **Required.**

**Outputs**

* **1**: The result of element-wise *Sinh* operation applied to the input tensor. A tensor of type *T* and the same shape as input tensor.

**Types**

* *T*: any supported numeric type.

**Example**

```xml
<layer ... type="Sinh">
    <input>
        <port id="0">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```
