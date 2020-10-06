## Acosh <a name="Acosh"></a> {#openvino_docs_ops_arithmetic_Acosh_1}

**Versioned name**: *Acosh-1*

**Category**: Arithmetic unary operation 

**Short description**: *Acosh* performs element-wise hyperbolic inverse cosine (arccosh) operation with given tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise acosh operation. A tensor of type T.

**Types**

* *T*: any numeric type.

*Acosh* does the following with the input tensor *a*:

\f[
a_{i} = acosh(a_{i})
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Acosh">
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
