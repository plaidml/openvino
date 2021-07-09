## Cosh <a name="Cosh"></a> {#openvino_docs_ops_arithmetic_Cosh_1}

**Versioned name**: *Cosh-1*

**Category**: Arithmetic unary operation

**Short description**: *Cosh* performs element-wise hyperbolic cosine operation on a given input tensor.

**Detailed description**: *Cosh* performs element-wise hyperbolic cosine (cosh) operation on a given input tensor, based on the following mathematical formula:

\f[
a_{i} = cosh(a_{i})
\f]

**Attributes**: *Cosh* operation has no attributes.

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise *Cosh* operation. A tensor of type *T* and the same shape as the input tensor.

**Types**

* *T*: any numeric type.

**Example**

```xml
<layer ... type="Cosh">
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
