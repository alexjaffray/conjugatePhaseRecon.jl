using BenchmarkTools
using Tullio
using Plots
using FFTW
using Zygote

plotlyjs()

## Naive version of explicit Forward operation
function Emul(x, shape, nodes::AbstractArray)

    K = size(nodes,2)
    x = reshape(x,shape)
    out = zeros(ComplexF64,K)
    X,Y = shape
    
    for nx = 1:X
        for ny = 1:Y
            for k  = 1:K
                out[k] += x[nx,ny] * exp(-2*1im*pi*(nodes[1,k]*(nx-X/2-1)+nodes[2,k]*(ny-Y/2-1)))
            end
        end
    end

    return vec(out)

end

function kernel(a)



end

function computePosForward(shape)

    X,Y = shape
    

    positions = Matrix{Float64}(undef,2,prod(shape))
    ox = ones(shape)
    oy = ones(shape)

    px = LinRange(-X/2,X/2-1,X).*ox
    py = LinRange(-Y/2,Y/2-1,Y).*oy

    positions[1,:] = vec(px)
    positions[2,:] = vec(py')

    return positions

end

function par_Emul(x,shape,nodes::AbstractArray, positions::AbstractArray)

    K = size(nodes,2)
    x = reshape(x,shape)
    y = zeros(ComplexF64,K)
    r = zeros(ComplexF64,K,K)
    X,Y = shape

    stride = 144
    nblocks = prod(shape)÷stride

    for k = 1:K
        for nx = 1:nblocks
            y[k] += vec(exp.(-2*1im*pi*nodes[:,k]'*positions[:,(nx-1)*stride .+ (1:stride)]))'*x[(nx-1)*stride .+ (1:stride)]
            r[k,:] = exp.(-2*1im*pi*nodes[:,k]'*positions[:,(nx-1)*stride .+ (1:stride)])
            
            @info positions
        end
    end

    return y,r

end

## Naive version of explicit adjoint op
function EHmul(x, shape, nodes::AbstractArray)

    K = size(nodes,2)
    out = zeros(ComplexF64,shape)
    X,Y = shape
    
    for nx = 1:X
        for ny = 1:Y
            for k  = 1:K
                out[nx,ny] += x[k] * conj(exp(-2*1im*pi*(nodes[1,k]*(nx-X/2-1)+nodes[2,k]*(ny-Y/2-1))))
            end
        end
    end

    return 1/prod(shape)*vec(out)

end

## Generate Cartesian Nodes
function cartesian2dNodes(::Type{T}, numProfiles, numSamplingPerProfile; kmin=(-0.5,-0.5), kmax=(0.5,0.5), kargs...) where T

    nodes = zeros(T, 2, numSamplingPerProfile, numProfiles)
    posX = collect( -ceil(Int64, (numSamplingPerProfile-1)/2.):floor(Int64, (numSamplingPerProfile-1)/2.) ) / numSamplingPerProfile
    posY = collect( -ceil(Int64, (numProfiles-1)/2.):floor(Int64, (numProfiles-1)/2.) ) / numProfiles

    for l = 1:numProfiles
        for k = 1:numSamplingPerProfile
            nodes[1,k,l] = posX[k]
            nodes[2,k,l] = posY[l]
        end
    end
    
    # rescale nodes according to the region covered by kmin & kmax
    nodes[1,:,:] .*= (kmax[1]-kmin[1])
    nodes[2,:,:] .*= (kmax[2]-kmin[2])
    
    # shift nodes to the region defined by kmin & kmax
    nodes[1,:,:] .+= 0.5*(kmin[1]+kmax[1])
    nodes[2,:,:] .+= 0.5*(kmin[2]+kmax[2])

    return reshape(nodes, 2, numSamplingPerProfile*numProfiles)

end

function jacobi(f, x)
    y, back = Zygote.pullback(f, x)
    back(1)[1], back(im)[1]
end
  
function wirtinger(f, x)
    du, dv = jacobi(f, x)
    (du' + im*dv')/2, (du + im*dv)/2
end

mag_test = x->(sinc(0.2*x))
phase_test = x->(0.01*x)
positions = LinRange(-12,11,12)

mag = mag_test.(positions) .* mag_test.(positions)'
phase = phase_test.(positions) .* phase_test.(positions)'

p1 = plot(mag)
p2 = plot(phase)

x = vec(mag .* exp.(1im .* 2 .*pi .*phase))

nodes = cartesian2dNodes(Float64,12,12)

shape = size(phase)

ft_x = Emul(x,shape,nodes)
x̂ = EHmul(ft_x,shape,nodes)

positions = computePosForward((12,12))

ft_x_par,E2 = par_Emul(x,shape,nodes,positions)

plot()