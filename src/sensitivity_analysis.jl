using BenchmarkTools
using Tullio

## Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
function EHmul(x, nodes::AbstractArray, positions::AbstractArray)

    @tullio EH[n, k] := exp <| (1.0im * pi * 2.0 * $positions[i, n] * nodes[i, k])
    @tullio y[n] := EH[n, k] * $x[k]

    return y

end

## Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
function EMul(x, nodes::AbstractArray, positions::AbstractArray)

    @tullio E[k, n] := exp <| (-1.0im * pi * 2.0 * nodes[i, k] * $positions[i, n])
    @tullio y[k] := E[k, n] * $x[n]

    return y

end

positions = LinRange(-64,63,128)
positions = vcat(positions',ones(size(positions))')
b = randn(ComplexF64,size(positions,2))
nodes = LinRange(-0.5,0.5,128)
nodes = vcat(nodes',ones(size(nodes))')

y = 1.0/sqrt(128) .* EMul(b,nodes,positions)
y2 = 1.0/sqrt(128) .* EHmul(y,nodes,positions)

