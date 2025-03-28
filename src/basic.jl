using LinearAlgebra
using .Memoize

export getH, getdH, getS, getdS, geteig, getdEs, getA, getdr, getvelocity, getspin, get_berry_curvature

const DEGEN_THRESH = [1.0e-4]

function set_degen_thresh(val)
    @warn "DEGEN_THRESH should be set before any calculations."
    DEGEN_THRESH[1] = val
    return nothing
end

const σ1 = [0 1; 1 0]
const σ2 = [0 -im; im 0]
const σ3 = [1 0; 0 -1]
const σs = [σ1, σ2, σ3]

@doc raw"""
```julia
getdH(tm::AbstractTBModel, order::Tuple{Int64,Int64,Int64},
    k::AbstractVector{<:Real})::Matrix{ComplexF64}
```

Calculate `order` derivative of Hamiltonian.
"""
@memoize k function getdH(tm::TBModel, order::Tuple{Int64,Int64,Int64}, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    dH = zeros(ComplexF64, tm.norbits, tm.norbits)
    for (R, hopping) in tm.hoppings
        Rc = tm.lat*R # R in Cartesian coordinate
        phase = exp(im*2π*(k⋅R))
        coeff = (im*Rc[1])^(order[1])*(im*Rc[2])^(order[2])*(im*Rc[3])^(order[3])*phase
        @. dH += coeff*hopping
    end
    return dH
end

@memoize k function getdH(
    sm::SharedTBModel,
    order::Tuple{Int64,Int64,Int64},
    k::AbstractVector{<:Real}
)::Matrix{ComplexF64}
    nhalfRs = (size(sm.Rs, 2) + 1) ÷ 2
    norbits = sm.norbits
    coeffs = im^sum(order) * prod(sm.Rcs.^order; dims=1) .* exp.(im * 2π * (k' * sm.Rs))
    tmp = reshape(sm.H * (coeffs[1, 1:nhalfRs]), (norbits, norbits))
    return tmp' + tmp
end


function _getdS(nm::TBModel, order::Tuple{Int64,Int64,Int64}, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    dS = zeros(ComplexF64, nm.norbits, nm.norbits)
    for (R, overlap) in nm.overlaps
        Rc = nm.lat*R # R in Cartesian coordinate
        phase = exp(im*2π*(k⋅R))
        coeff = (im*Rc[1])^(order[1])* (im*Rc[2])^(order[2])*(im*Rc[3])^(order[3])*phase
        @. dS += coeff*overlap
    end
    return dS
end

@doc raw"""
```julia
getdS(tm::AbstractTBModel, order::Tuple{Int64,Int64,Int64},
    k::AbstractVector{<:Real})::Matrix{ComplexF64}
```

Calculate `order` derivative of overlap.
"""
@memoize k function getdS(tm::TBModel, order::Tuple{Int64,Int64,Int64}, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    return tm.isorthogonal ? (order == (0, 0, 0) ? I(tm.norbits) : 0I(tm.norbits)) : _getdS(tm, order, k)
end

function _getdS(
    sm::SharedTBModel,
    order::Tuple{Int64,Int64,Int64},
    k::AbstractVector{<:Real}
)::Matrix{ComplexF64}
    nhalfRs = (size(sm.Rs, 2) + 1) ÷ 2
    norbits = sm.norbits
    coeffs = im^sum(order) * prod(sm.Rcs.^order; dims=1) .* exp.(im * 2π * (k' * sm.Rs))
    tmp = reshape(sm.S * (coeffs[1, 1:nhalfRs]), (norbits, norbits))
    return tmp' + tmp
end

@memoize k function getdS(sm::SharedTBModel, order::Tuple{Int64,Int64,Int64}, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    return sm.isorthogonal ? (order == (0, 0, 0) ? I(sm.norbits) : 0I(sm.norbits)) : _getdS(sm, order, k)
end


@doc raw"""
```julia
getdAw(tm::AbstractTBModel, α::Int64, order::Tuple{Int64,Int64,Int64},
    k::AbstractVector{<:Real})::Matrix{ComplexF64}
```

Calculate `order` derivative of ``i⟨u_n^{(W)}|∂_{k_α}u_m^{(W)}⟩``.
"""
@memoize k function getdAw(tm::TBModel, α::Int64, order::Tuple{Int64,Int64,Int64},
    k::AbstractVector{<:Real})::Matrix{ComplexF64}
    dAw = zeros(ComplexF64, tm.norbits, tm.norbits)
    for (R, pos) in tm.positions
        Rc = tm.lat*R
        phase = exp(im*2π*(k⋅R))
        coeff = (im*Rc[1])^(order[1])*(im*Rc[2])^(order[2])*(im*Rc[3])^(order[3])*phase
        @. dAw += coeff*pos[α]
    end
    return dAw'
end

@memoize k function getdAw(
    sm::SharedTBModel,
    α::Int64,
    order::Tuple{Int64,Int64,Int64},
    k::AbstractVector{<:Real}
)::Matrix{ComplexF64}
    nhalfRs = (size(sm.Rs, 2) + 1) ÷ 2
    norbits = sm.norbits
    coeffs = im^sum(order) * prod(sm.Rcs.^order; dims=1) .* exp.(im * 2π * (k' * sm.Rs))
    dAw = reshape(sm.r[α] * (coeffs[1, :]), (norbits, norbits))
    return dAw'
end


@doc raw"""
```julia
getH(tm::AbstractTBModel, k::AbstractVector{<:Real})::Matrix{ComplexF64}
```

Calculate Hamiltonian at a reduced `k` point.
"""
function getH(tm::AbstractTBModel, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    return getdH(tm, (0, 0, 0), k)
end


@doc raw"""
```julia
getS(tm::AbstractTBModel, k::AbstractVector{<:Real})::Matrix{ComplexF64}
```

Calculate overlap matrix at a reduced `k` point.
"""
function getS(tm::AbstractTBModel, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    return getdS(tm, (0, 0, 0), k)
end


"""
HermEig wraps eigenvalues and eigenvectors of a Hermitian eigenvalue problem.

# Fields
 - `values::Vector{Float64}`: eigenvalues
 - `vectors::Matrix{ComplexF64}`: eigenvectors stored in column
"""
struct HermEig
    values::Vector{Float64}
    vectors::Matrix{ComplexF64}
end

Base.iterate(S::HermEig) = (S.values, Val(:vectors))
Base.iterate(S::HermEig, ::Val{:vectors}) = (S.vectors, Val(:done))
Base.iterate(S::HermEig, ::Val{:done}) = nothing

@doc raw"""
```julia
geteig(tm::AbstractTBModel, k::AbstractVector{<:Real})::HermEig
```

Calculate eigenvalues and eigenvectors of `tm` at a reduced `k` point.
"""
@memoize k function geteig(tm::AbstractTBModel, k::AbstractVector{<:Real})::HermEig
    H = getH(tm, k)
    if tm.isorthogonal
        (Es, V) = eigen(Hermitian(H))
    else
        S = getS(tm, k)
        (Es, V) = eigen(Hermitian(H), Hermitian(S))
    end
    return HermEig(Es, V)
end


@doc raw"""
```julia
getAw(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
```

Calculate ``i⟨u_n^{(W)}|∂_{k_α}u_m^{(W)}⟩``.
"""
function getAw(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    return getdAw(tm, α, (0, 0, 0), k)
end


@memoize k function getdHbar(tm::AbstractTBModel, order::Tuple{Int64,Int64,Int64}, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    dH = getdH(tm, order, k)
    V = geteig(tm, k).vectors
    return V'*dH*V
end


@memoize k function getdSbar(tm::AbstractTBModel, order::Tuple{Int64,Int64,Int64}, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    dS = getdS(tm, order, k)
    V = geteig(tm, k).vectors
    return V'*dS*V
end

@memoize k function getdAwbar(tm::AbstractTBModel, α::Int64, order::Tuple{Int64,Int64,Int64}, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    dAw = getdAw(tm, α, order, k)
    V = geteig(tm, k).vectors
    return V'*dAw*V
end


function getAwbar(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    return getdAwbar(tm, α, (0, 0, 0), k)
end


function getorder(α::Int64)::Tuple{Int64, Int64, Int64}
    order = zeros(Int64, 3)
    order[α] += 1
    return Tuple(order)
end

function getorder(α::Int64, β::Int64)::Tuple{Int64, Int64, Int64}
    order = zeros(Int64, 3)
    order[α] += 1; order[β] += 1
    return Tuple(order)
end

function getorder(α::Int64, β::Int64,γ::Int64)::Tuple{Int64, Int64, Int64}
    order = zeros(Int64, 3)
    order[α] += 1; order[β] += 1; order[γ] += 1
    return Tuple(order)
end


@memoize k function getD(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    order = getorder(α)
    dHbar = getdHbar(tm, order, k); dSbar = getdSbar(tm, order, k)
    Es = geteig(tm, k).values
    D = zeros(ComplexF64, tm.norbits, tm.norbits)
    Awbar = getAwbar(tm, α, k)
    for m in 1:tm.norbits, n in 1:tm.norbits
        En = Es[n]; Em = Es[m]
        if abs(En-Em) > DEGEN_THRESH[1]
            D[n, m] = (dHbar[n, m]-Em*dSbar[n, m])/(Em-En)
        else
            D[n, m] = im*Awbar[n,m]
        end
    end
    return D
end


@memoize k function getHαnm(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    order = getorder(α)
    dHbar = getdHbar(tm, order, k); dSbar = getdSbar(tm, order, k)
    Es = geteig(tm, k).values
    H = zeros(ComplexF64, tm.norbits, tm.norbits)
    Dα=getD(tm,α,k)
    Awbar = getAwbar(tm, α, k)
    for m in 1:tm.norbits, n in 1:tm.norbits
        En = Es[n]; Em = Es[m]
        H[n, m] = (dHbar[n, m]-Em*dSbar[n, m]-Dα[n,m]*Em+En*Dα[n,m])
    end
    return H
end

@memoize k function getD2(tm::AbstractTBModel, α::Int64,β::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    Es, _ = geteig(tm, k)
    m = 0  # 初始化 m
    n = 0  # 初始化 n
    dHαbar = getdHbar(tm, getorder(α), k)
    dHβbar = getdHbar(tm, getorder(β), k)
    dSαbar = getdSbar(tm, getorder(α), k)
    dSβbar = getdSbar(tm, getorder(β), k)
    dHαβbar = getdHbar(tm, getorder(α, β), k)
    dSαβbar = getdSbar(tm, getorder(α, β), k)
    Dα = getD(tm, α, k)
    Dβ = getD(tm, β, k)
    dEs = zeros(tm.norbits)
    foo1 = dHαbar * Dβ + dHβbar * Dα
    foo2 = dSαbar * Dβ + dSβbar * Dα
    foo3α = getdEs(tm, α, k)
    foo3β = getdEs(tm, β, k) 
    foo4αβ = Dα*Dβ
    foo4βα = Dβ*Dα
    foo5αβ = zeros(ComplexF64, tm.norbits, tm.norbits)
    foo5βα = zeros(ComplexF64, tm.norbits, tm.norbits)
    D2 = zeros(ComplexF64, tm.norbits, tm.norbits)
    Awαβ = getdAwbar(tm, α, getorder(β), k)
    Awα = getAwbar(tm, α, k)
    L=Dβ*Awα-Awα*Dβ
    for m in 1:tm.norbits, n in 1:tm.norbits
        En = Es[n]; Em = Es[m]
        foo5αβ[n,m]=foo3α[m]*Dβ[n,m]
        foo5βα[n,m]=foo3β[m]*Dα[n,m]
        if abs(En-Em) > DEGEN_THRESH[1]
            D2[n, m] += (dHαβbar[n, m] - dSαβbar[n, m] * Em)/(Em-En)
            D2[n, m] += (foo1[n, m])/(Em-En)
            D2[n, m] -= (foo2[n, m] * Em)/(Em-En)
            D2[n, m] -= (dSαbar[n,m] * foo3β[m] + dSβbar[n, m] * foo3α[m])/(Em-En)
            D2[n, m] -= (foo5αβ[n,m]+foo5βα[n,m])/(Em-En)
        else
            D2[n, m] = 0
        end
    end
    return D2
end

@memoize k function getDl(tm::AbstractTBModel, α::Int64,β::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    Es, _ = geteig(tm, k)
    m = 0  # 初始化 m
    n = 0  # 初始化 n
    eβα=gete(tm,β,α,k)
    eαβ=gete(tm,α,β,k)
    eab=gete2(tm,α,β,k)
    Dl = zeros(ComplexF64, tm.norbits, tm.norbits)
    for m in 1:tm.norbits, n in 1:tm.norbits
        En = Es[n]; Em = Es[m]
        if abs(En-Em) > DEGEN_THRESH[1]
            Dl[n, m] += eab[n,m]
            Dl[n, m] += eαβ[n,m]
            Dl[n, m] += eβα[n,m]
        else
            Dl[n, m] = 0
        end
    end
    return Dl
end
"""
```julia
getdEs(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real})
-->dEs::Vector{Float64}
```

Calculate dE/dk for `tm` at `k` in the `α` direction.

Calculation method is provided in [Wang et al, 2019]. The relevant equation
is Eq. (13). Although in that equation, there is no energy different denominator,
it is still implicitly assumed that the band is nondegenerate. Therefore, dEs[n]
is only correct if n is nondegenerate or completely degenerate.

This function is memoized, which means the arguments and results of the function should
never be modified.
"""
@memoize k function getdEs(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real})::Vector{Float64}
    order = getorder(α)
    dHbar = getdHbar(tm, order, k)
    dSbar = getdSbar(tm, order, k)
    Es = geteig(tm, k).values
    dEs = zeros(tm.norbits)
    for n in 1:tm.norbits
        dEs[n] = real(dHbar[n, n]-Es[n]*dSbar[n, n])
    end
    return dEs
end

@doc raw"""
```julia
getdEs(
    tm::AbstractTBModel,
    α::Integer,
    β::Integer,
    k::AbstractVector{<:Real}
) => dEs::Vector{Float64}
```

Calculate d^2 E / dkα dkβ for `tm` at `k`. `α` and `β` are Cartesian directions.

dEs[n] is only correct if n is nondegenerate or completely degenerate.

This function is memoized, which means the arguments and results of the function should
never be modified.
"""
@memoize k function getdEs(
    tm::AbstractTBModel,
    α::Integer,
    β::Integer,
    k::AbstractVector{<:Real}
)::Vector{Float64}
    Es, _ = geteig(tm, k)
    dHαbar = getdHbar(tm, getorder(α), k)
    dHβbar = getdHbar(tm, getorder(β), k)
    dSαbar = getdSbar(tm, getorder(α), k)
    dSβbar = getdSbar(tm, getorder(β), k)
    dHαβbar = getdHbar(tm, getorder(α, β), k)
    dSαβbar = getdSbar(tm, getorder(α, β), k)
    Dα = getD(tm, α, k)
    Dβ = getD(tm, β, k)
    dEs = zeros(tm.norbits)
    foo1 = dHαbar * Dβ + dHβbar * Dα
    foo2 = dSαbar * Dβ + dSβbar * Dα
    foo3α = getdEs(tm, α, k)
    foo3β = getdEs(tm, β, k)
    for n in 1:tm.norbits
        dEs[n] += real(dHαβbar[n, n] - dSαβbar[n, n] * Es[n])
        dEs[n] += real(foo1[n, n])
        dEs[n] -= real(foo2[n, n] * Es[n])
        dEs[n] -= real(dSαbar[n, n] * foo3β[n] + dSβbar[n, n] * foo3α[n])
        dEs[n] -= real(Dα[n, n] * foo3β[n] + Dβ[n, n] * foo3α[n])
    end
    return dEs
end

@doc raw"""
```julia
getdEs(
    tm::AbstractTBModel,
    α::Integer,
    β::Integer,
    k::AbstractVector{<:Real}
) => dEs::Vector{Float64}
```

Calculate d^2 E / dkα dkβ for `tm` at `k`. `α` and `β` are Cartesian directions.

dEs[n,m] is only correct if n is nondegenerate or completely degenerate.

This function is memoized, which means the arguments and results of the function should
never be modified.
"""
@memoize k function getdHs(
    tm::AbstractTBModel,
    α::Integer,
    β::Integer,
    k::AbstractVector{<:Real}
)::Matrix{ComplexF64}
    Es, _ = geteig(tm, k)
    dHαbar = getdHbar(tm, getorder(α), k)
    dHβbar = getdHbar(tm, getorder(β), k)
    dSαbar = getdSbar(tm, getorder(α), k)
    dSβbar = getdSbar(tm, getorder(β), k)
    dHαβbar = getdHbar(tm, getorder(α, β), k)
    dSαβbar = getdSbar(tm, getorder(α, β), k)
    Dα = getD(tm, α, k)
    Dβ = getD(tm, β, k)
    dEs = zeros(ComplexF64, tm.norbits, tm.norbits)
    foo1 = dHαbar * Dβ + dHβbar * Dα
    foo2 = dSαbar * Dβ + dSβbar * Dα
    foo3α = getdEs(tm, α, k)
    foo3β = getdEs(tm, β, k)
    for m in 1:tm.norbits, n in 1:tm.norbits
            dEs[n,m] += dHαβbar[n, m] - dSαβbar[n, m] * Es[m]
            dEs[n,m] += foo1[n, m]
            dEs[n,m] -= foo2[n, m] * Es[m]
            dEs[n,m] -= dSαbar[n, m] * foo3β[m] + dSβbar[n, m] * foo3α[m]
            dEs[n,m] -= Dα[n, m]*foo3β[n]  + Dβ[n, m]*foo3α[n]
    end
    return dEs
end


@doc raw"""
```julia
getdEs(
    tm::AbstractTBModel,
    α::Integer,
    β::Integer,
    k::AbstractVector{<:Real}
) => dEs::Vector{Float64}
```

Calculate d^3 E / dkα dkβ dkγ for `tm` at `k`. `α` and `β``γ` are Cartesian directions.

dEs[n] is only correct if n is nondegenerate or completely degenerate.

This function is memoized, which means the arguments and results of the function should
never be modified.
"""


@memoize k function getdEs(
    tm::AbstractTBModel,
    α::Integer,
    β::Integer,
    γ::Integer,    
    k::AbstractVector{<:Real}
)::Vector{Float64}
    Es, _ = geteig(tm, k)
    dHαbar = getdHbar(tm, getorder(α), k)
    dHβbar = getdHbar(tm, getorder(β), k)
    dHγbar = getdHbar(tm, getorder(γ), k)
    dSαbar = getdSbar(tm, getorder(α), k)
    dSβbar = getdSbar(tm, getorder(β), k)
    dSγbar = getdSbar(tm, getorder(γ), k)
    dHαβbar = getdHbar(tm, getorder(α, β), k)
    dHβγbar = getdHbar(tm, getorder(β,γ), k)
    dHγαbar = getdHbar(tm, getorder(γ,α), k)
    dSαβbar = getdSbar(tm, getorder(α, β), k)
    dSβγbar = getdSbar(tm, getorder(β,γ), k)
    dSγαbar = getdSbar(tm, getorder(γ,α), k)
    dHαβγbar = getdHbar(tm, getorder(α, β, γ), k)
    dSαβγbar = getdSbar(tm, getorder(α, β, γ), k)
    Dα = getD(tm, α, k)
    Dβ = getD(tm, β, k)
    Dγ = getD(tm, γ, k)
    dEs = zeros(tm.norbits)
    foo1 = (dHαβbar * Dγ + dHβγbar * Dα +dHγαbar * Dβ) +(dHαbar*Dβ*Dγ+dHαbar*Dγ*Dβ+dHβbar*Dα*Dγ+dHβbar*Dγ*Dα + dHγbar*Dα*Dβ + dHγbar*Dβ*Dα)*1/2
    foo2 = (dSαβbar * Dγ + dSβγbar * Dα +dSγαbar * Dβ) +(dSαbar*Dβ*Dγ+dSαbar*Dγ*Dβ+dSβbar*Dα*Dγ+dSβbar*Dγ*Dα + dSγbar*Dα*Dβ + dSγbar*Dβ*Dα)*1/2
    foo3α = getdEs(tm, α, k)
    foo3β = getdEs(tm, β, k)
    foo3γ = getdEs(tm, γ, k)
    foo3αβ = getdEs(tm, α, β, k)
    foo3βγ = getdEs(tm, β, γ, k)
    foo3γα = getdEs(tm, γ, α, k)
    foo4αβ = dSαbar*Dβ
    foo4αγ = dSαbar*Dγ
    foo4βα = dSβbar*Dα
    foo4βγ = dSβbar*Dγ
    foo4γα = dSγbar*Dα
    foo4γβ = dSγbar*Dβ
    foo5αβ = Dα*Dβ
    foo5αγ = Dα*Dγ
    foo5βα = Dβ*Dα
    foo5βγ = Dβ*Dγ
    foo5γα = Dγ*Dα
    foo5γβ = Dγ*Dβ
    for n in 1:tm.norbits
        dEs[n] += real(dHαβγbar[n, n] - dSαβγbar[n, n] * Es[n])
        dEs[n] += real(foo1[n, n])
        dEs[n] -= real(foo2[n, n] * Es[n])
        dEs[n] -= real((dSαβbar[n, n] * foo3γ[n] + dSβγbar[n, n] * foo3α[n]+dSγαbar[n, n] * foo3β[n]))
        dEs[n] -= real((dSγbar[n,n]*foo3αβ[n] + dSαbar[n,n]*foo3βγ[n]+dSβbar[n,n]*foo3γα[n])+(dSαbar[n,n]*foo3β[n]*Dγ[n, n]+dSαbar[n,n]*foo3γ[n]*Dβ[n, n] + dSγbar[n,n]*foo3α[n]*Dβ[n, n]+dSγbar[n,n]*foo3β[n]*Dα[n, n]+dSβbar[n,n]*foo3γ[n]*Dα[n, n]+dSβbar[n,n]*foo3α[n]*Dγ[n, n]))
        dEs[n] -=real((Dγ[n,n]*foo3αβ[n] + Dα[n,n]*foo3βγ[n]+Dβ[n,n]*foo3γα[n])+(Dα[n,n]*foo3β[n]*Dγ[n, n]+Dα[n,n]*foo3γ[n]*Dβ[n, n] + Dγ[n,n]*foo3α[n]*Dβ[n, n]+Dγ[n,n]*foo3β[n]*Dα[n, n]+Dβ[n,n]*foo3γ[n]*Dα[n, n]+Dβ[n,n]*foo3α[n]*Dγ[n, n])*1/2)
    end
    return dEs
end  

@memoize k function getdEs2(
    tm::AbstractTBModel,
    α::Integer,
    β::Integer,
    γ::Integer,    
    k::AbstractVector{<:Real}
)::Vector{Float64}
    Es, _ = geteig(tm, k)
    dHαbar = getdHbar(tm, getorder(α), k)
    dHβbar = getdHbar(tm, getorder(β), k)
    dHγbar = getdHbar(tm, getorder(γ), k)
    dSαbar = getdSbar(tm, getorder(α), k)
    dSβbar = getdSbar(tm, getorder(β), k)
    dSγbar = getdSbar(tm, getorder(γ), k)
    dHαβbar = getdHbar(tm, getorder(α, β), k)
    dHβγbar = getdHbar(tm, getorder(β,γ), k)
    dHγαbar = getdHbar(tm, getorder(γ,α), k)
    dSαβbar = getdSbar(tm, getorder(α, β), k)
    dSβγbar = getdSbar(tm, getorder(β,γ), k)
    dSγαbar = getdSbar(tm, getorder(γ,α), k)
    dHαβγbar = getdHbar(tm, getorder(α, β, γ), k)
    dSαβγbar = getdSbar(tm, getorder(α, β, γ), k)
    Dα = getD(tm, α, k)
    Dβ = getD(tm, β, k)
    Dγ = getD(tm, γ, k)
    Dαβ = getDl(tm,α,β,k)
    Dβγ = getDl(tm,β,γ,k)
    Dγα = getDl(tm,γ,α,k)
    dEs = zeros(tm.norbits)
    foo1 = (dHαβbar * Dγ + dHβγbar * Dα +dHγαbar * Dβ) +(-dHαbar*Dβ*Dγ+dHαbar*Dβγ+dHαbar*Dγ*Dβ+dHβbar*Dα*Dγ+dHβbar*Dγα-dHβbar*Dγ*Dα - dHγbar*Dα*Dβ +dHγbar*Dαβ+ dHγbar*Dβ*Dα)
    foo2 = (dSαβbar * Dγ + dSβγbar * Dα +dSγαbar * Dβ) +(dSαbar*Dβ*Dγ+dSαbar*Dγ*Dβ+dSβbar*Dα*Dγ+dSβbar*Dγ*Dα + dSγbar*Dα*Dβ + dSγbar*Dβ*Dα)*1/2
    foo3α = getdEs(tm, α, k)
    foo3β = getdEs(tm, β, k)
    foo3γ = getdEs(tm, γ, k)
    foo3αβ = getdEs(tm, α, β, k)
    foo3βγ = getdEs(tm, β, γ, k)
    foo3γα = getdEs(tm, γ, α, k)
    foo4αβ = dSαbar*Dβ
    foo4αγ = dSαbar*Dγ
    foo4βα = dSβbar*Dα
    foo4βγ = dSβbar*Dγ
    foo4γα = dSγbar*Dα
    foo4γβ = dSγbar*Dβ
    foo5αβ = Dα*Dβ
    foo5αγ = Dα*Dγ
    foo5βα = Dβ*Dα
    foo5βγ = Dβ*Dγ
    foo5γα = Dγ*Dα
    foo5γβ = Dγ*Dβ
    for n in 1:tm.norbits
        dEs[n] += real(dHαβγbar[n, n] - dSαβγbar[n, n] * Es[n])
        dEs[n] += real(foo1[n, n])
        dEs[n] -= real(foo2[n, n] * Es[n])
        dEs[n] -= real((dSαβbar[n, n] * foo3γ[n] + dSβγbar[n, n] * foo3α[n]+dSγαbar[n, n] * foo3β[n]))
        dEs[n] -= real((dSγbar[n,n]*foo3αβ[n] + dSαbar[n,n]*foo3βγ[n]+dSβbar[n,n]*foo3γα[n])+(dSαbar[n,n]*foo3β[n]*Dγ[n, n]+dSαbar[n,n]*foo3γ[n]*Dβ[n, n] + dSγbar[n,n]*foo3α[n]*Dβ[n, n]+dSγbar[n,n]*foo3β[n]*Dα[n, n]+dSβbar[n,n]*foo3γ[n]*Dα[n, n]+dSβbar[n,n]*foo3α[n]*Dγ[n, n]))
        dEs[n] -=real((foo3αβ[n]*Dγ[n,n] +foo3βγ[n]* Dα[n,n]+foo3γα[n]*Dβ[n,n])+(-foo3α[n]*foo5βγ[n,n]+foo3α[n]*Dβγ[n,n]+foo3α[n]*foo5γβ[n,n]+foo3β[n]*foo5αγ[n,n]+foo3β[n]*Dγα[n,n]-foo3β[n]*foo5γα[n,n] - foo3γ[n]*foo5αβ[n,n] +foo3γ[n]*Dαβ[n,n]+ foo3γ[n]*foo5βα[n,n]))
    end
    return dEs
end  

@memoize k function getdEs3(
    tm::AbstractTBModel,
    α::Integer,
    β::Integer,
    γ::Integer,    
    k::AbstractVector{<:Real}
)::Vector{Float64}
    Es, _ = geteig(tm, k)
    dHαbar = getdHbar(tm, getorder(α), k)
    dHβbar = getdHbar(tm, getorder(β), k)
    dHγbar = getdHbar(tm, getorder(γ), k)
    dSαbar = getdSbar(tm, getorder(α), k)
    dSβbar = getdSbar(tm, getorder(β), k)
    dSγbar = getdSbar(tm, getorder(γ), k)
    dHαβbar = getdHbar(tm, getorder(α, β), k)
    dHβγbar = getdHbar(tm, getorder(β,γ), k)
    dHγαbar = getdHbar(tm, getorder(γ,α), k)
    dSαβbar = getdSbar(tm, getorder(α, β), k)
    dSβγbar = getdSbar(tm, getorder(β,γ), k)
    dSγαbar = getdSbar(tm, getorder(γ,α), k)
    dHαβγbar = getdHbar(tm, getorder(α, β, γ), k)
    dSαβγbar = getdSbar(tm, getorder(α, β, γ), k)
    Dα = getD(tm, α, k)
    Dβ = getD(tm, β, k)
    Dγ = getD(tm, γ, k)
    Dαβ = getDl(tm,α,β,k)
    Dβγ = getDl(tm,β,γ,k)
    Dγα = getDl(tm,γ,α,k)
    Dβα = getDl(tm,β,α,k)
    Dγβ = getDl(tm,γ,β,k)
    Dαγ = getDl(tm,α,γ,k)
    dEs = zeros(tm.norbits)
    foo1 = (dHαβbar * Dγ + dHβγbar * Dα +dHγαbar * Dβ) +(dHαbar*Dβγ+dHβbar*Dγα+dHγbar*Dαβ+dHαbar*Dγβ+dHβbar*Dαγ+dHγbar*Dβα)*1/2
    foo2 = (dSαβbar * Dγ + dSβγbar * Dα +dSγαbar * Dβ) +(dSαbar*Dβγ+dSαbar*Dγβ+dSβbar*Dαγ+dSβbar*Dγα+dSγbar*Dαβ+dSγbar*Dβα)*1/2
    foo3α = getdEs(tm, α, k)
    foo3β = getdEs(tm, β, k)
    foo3γ = getdEs(tm, γ, k)
    foo3αβ = getdEs(tm, α, β, k)
    foo3βγ = getdEs(tm, β, γ, k)
    foo3γα = getdEs(tm, γ, α, k)
    foo4αβ = dSαbar*Dβ
    foo4αγ = dSαbar*Dγ
    foo4βα = dSβbar*Dα
    foo4βγ = dSβbar*Dγ
    foo4γα = dSγbar*Dα
    foo4γβ = dSγbar*Dβ
    foo5αβ = Dα*Dβ
    foo5αγ = Dα*Dγ
    foo5βα = Dβ*Dα
    foo5βγ = Dβ*Dγ
    foo5γα = Dγ*Dα
    foo5γβ = Dγ*Dβ
    for n in 1:tm.norbits
        dEs[n] += real(dHαβγbar[n, n] - dSαβγbar[n, n] * Es[n])
        dEs[n] += real(foo1[n, n])
        dEs[n] -= real(foo2[n, n] * Es[n])
        dEs[n] -= real((dSαβbar[n, n] * foo3γ[n] + dSβγbar[n, n] * foo3α[n]+dSγαbar[n, n] * foo3β[n]))
        dEs[n] -= real((dSγbar[n,n]*foo3αβ[n] + dSαbar[n,n]*foo3βγ[n]+dSβbar[n,n]*foo3γα[n])+(dSαbar[n,n]*foo3β[n]*Dγ[n, n]+dSαbar[n,n]*foo3γ[n]*Dβ[n, n] + dSγbar[n,n]*foo3α[n]*Dβ[n, n]+dSγbar[n,n]*foo3β[n]*Dα[n, n]+dSβbar[n,n]*foo3γ[n]*Dα[n, n]+dSβbar[n,n]*foo3α[n]*Dγ[n, n]))
        dEs[n] -=real((foo3αβ[n]*Dγ[n,n] +foo3βγ[n]* Dα[n,n]+foo3γα[n]*Dβ[n,n])+(foo3α[n]*Dβγ[n,n]+foo3β[n]*Dγα[n,n]+foo3γ[n]*Dαβ[n,n]+foo3α[n]*Dγβ[n,n]+foo3β[n]*Dαγ[n,n]+foo3γ[n]*Dβα[n,n])*1/2)
    end
    return dEs
end  
@memoize k function getdHs(
    tm::AbstractTBModel,
    α::Integer,
    β::Integer,
    γ::Integer,    
    k::AbstractVector{<:Real}
)::Matrix{ComplexF64}
    Es, _ = geteig(tm, k)
    dHαbar = getdHbar(tm, getorder(α), k)
    dHβbar = getdHbar(tm, getorder(β), k)
    dHγbar = getdHbar(tm, getorder(γ), k)
    dSαbar = getdSbar(tm, getorder(α), k)
    dSβbar = getdSbar(tm, getorder(β), k)
    dSγbar = getdSbar(tm, getorder(γ), k)
    dHαβbar = getdHbar(tm, getorder(α, β), k)
    dHβγbar = getdHbar(tm, getorder(β,γ), k)
    dHγαbar = getdHbar(tm, getorder(γ,α), k)
    dSαβbar = getdSbar(tm, getorder(α, β), k)
    dSβγbar = getdSbar(tm, getorder(β,γ), k)
    dSγαbar = getdSbar(tm, getorder(γ,α), k)
    dHαβγbar = getdHbar(tm, getorder(α, β, γ), k)
    dSαβγbar = getdSbar(tm, getorder(α, β, γ), k)
    Dα = getD(tm, α, k)
    Dβ = getD(tm, β, k)
    Dγ = getD(tm, γ, k)
    Dαβ = getDl(tm,α,β,k)
    Dβγ = getDl(tm,β,γ,k)
    Dγα = getDl(tm,γ,α,k)
    dEs = zeros(ComplexF64, tm.norbits, tm.norbits)
    foo1 = (dHαβbar * Dγ + dHβγbar * Dα +dHγαbar * Dβ) +(-dHαbar*Dβ*Dγ+dHαbar*Dβγ+dHαbar*Dγ*Dβ+dHβbar*Dα*Dγ+dHβbar*Dγα-dHβbar*Dγ*Dα - dHγbar*Dα*Dβ +dHγbar*Dαβ+ dHγbar*Dβ*Dα)
    foo2 = (dSαβbar * Dγ + dSβγbar * Dα +dSγαbar * Dβ) +(dSαbar*Dβ*Dγ+dSαbar*Dγ*Dβ+dSβbar*Dα*Dγ+dSβbar*Dγ*Dα + dSγbar*Dα*Dβ + dSγbar*Dβ*Dα)*1/2
    foo3α = getdEs(tm, α, k)
    foo3β = getdEs(tm, β, k)
    foo3γ = getdEs(tm, γ, k)
    foo3αβ = getdEs(tm, α, β, k)
    foo3βγ = getdEs(tm, β, γ, k)
    foo3γα = getdEs(tm, γ, α, k)
    foo4αβ = dSαbar*Dβ
    foo4αγ = dSαbar*Dγ
    foo4βα = dSβbar*Dα
    foo4βγ = dSβbar*Dγ
    foo4γα = dSγbar*Dα
    foo4γβ = dSγbar*Dβ
    foo5αβ = Dα*Dβ
    foo5αγ = Dα*Dγ
    foo5βα = Dβ*Dα
    foo5βγ = Dβ*Dγ
    foo5γα = Dγ*Dα
    foo5γβ = Dγ*Dβ
    for m in 1:tm.norbits, n in 1:tm.norbits
        dEs[n,m] += dHαβγbar[n, m] - dSαβγbar[n, m] * Es[m]
        dEs[n,m] += foo1[n, m]
        dEs[n,m] -= foo2[n, m] * Es[m]
        dEs[n,m] -= (dSαβbar[n, m] * foo3γ[m] + dSβγbar[n, m] * foo3α[m]+dSγαbar[n, m] * foo3β[m])
        dEs[n,m] -= (dSγbar[n,m]*foo3αβ[m] + dSαbar[n,m]*foo3βγ[m]+dSβbar[n,m]*foo3γα[m])+(dSαbar[n,m]*foo3β[m]*Dγ[m, m]+dSαbar[n,m]*foo3γ[m]*Dβ[m, m]+ dSγbar[n,m]*foo3α[m]*Dβ[m, m]+dSγbar[n,m]*foo3β[m]*Dα[m, m]+dSβbar[n,m]*foo3γ[m]*Dα[m, m]+dSβbar[n,m]*foo3α[m]*Dγ[m, m])
        dEs[n,m] -=(foo3αβ[n]*Dγ[n,m] +foo3βγ[n]* Dα[n,m]+foo3γα[n]*Dβ[n,m])-foo3α[n]*foo5βγ[n,m]+foo3α[n]*Dβγ[n,m]+foo3α[n]*foo5γβ[n,m]+foo3β[n]*foo5αγ[n,m]+foo3β[n]*Dγα[n,m]-foo3β[n]*foo5γα[n,m] - foo3γ[n]*foo5αβ[n,m] +foo3γ[n]*Dαβ[n,m]+ foo3γ[n]*foo5βα[n,m]
    end
    return dEs
end   

@memoize k function getDHs(
    tm::AbstractTBModel,
    α::Integer,
    β::Integer,
    γ::Integer,    
    k::AbstractVector{<:Real}
)::Matrix{ComplexF64}
    dHαβγbar=getdHs(tm, α, β, γ, k)
    dEs = zeros(ComplexF64, tm.norbits, tm.norbits)
    Dγ = getD(tm, γ, k)
    dHαβbar=getdHs(tm, α, β, k)
    foo1=Dγ*dHαβbar-dHαβbar*Dγ
    for m in 1:tm.norbits, n in 1:tm.norbits
        dEs[n,m] += real(dHαβγbar[n, m])
        dEs[n,m] -= real(foo1[n, m])
    end
    return dEs
end       
"""

```julia
getA(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real})
-->A::Matrix{ComplexF64}
```

Calculate Berry connection ``i⟨u_n|∂_α|u_m⟩`` for `tm` at `k`.

Calculation method is provided in [Wang et al, 2019]. The relevant equation
is Eq. (14). Since Eq. (14) assumes band m is nondegenerate, A[n, m] is only
correct if m is nondegenerate or completely degenerate.
"""
@memoize k function getA(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    A = im*getD(tm, α, k) + getAwbar(tm, α, k)
    return A
end


"""
```julia
function getdr(tm::AbstractTBModel, α::Int64, β::Int64, k::AbstractVector{<:Real})
-->dr::Matrix{ComplexF64}
```

Compute ``∂_β r_α`` for `tm` at `k`. r[n, m] = A[n, m] if n != m and r[n, n] = 0.

dr is calculated by directly differentiating Eq. (14) of [Wang et al, 2019].
dr[n, m] is only correct when (i) both band n and band m are nondegenerate or
(ii) both band n and band m are completely degenerate but ``E_n≠E_m``.
"""
@memoize k function getdr(tm::AbstractTBModel, α::Int64, β::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    orderα = getorder(α); orderβ = getorder(β); orderαβ = getorder(α, β)
    dHαbar = getdHbar(tm, orderα, k); dHαβbar = getdHbar(tm, orderαβ, k)
    dSαbar = getdSbar(tm, orderα, k); dSαβbar = getdSbar(tm, orderαβ, k)
    Awαbar = getAwbar(tm, α, k); dAwαβbar = getdAwbar(tm, α, orderβ, k)
    Es = geteig(tm, k).values
    dEs = getdEs(tm, β, k)
    Dα = getD(tm, α, k); Dβ = getD(tm, β, k)
    dr = zeros(ComplexF64, tm.norbits, tm.norbits)
    tmpH = dHαbar*Dβ+dHαβbar+Dβ'*dHαbar; tmpS = dSαbar*Dβ+dSαβbar+Dβ'*dSαbar
    tmpA = Awαbar*Dβ+dAwαβbar+Dβ'*Awαbar
    for m in 1:tm.norbits, n in 1:tm.norbits
        En = Es[n]; Em = Es[m]; dEn = dEs[n]; dEm = dEs[m]
        if abs(En-Em) > DEGEN_THRESH[1]
            dr[n, m] += im*tmpH[n, m]/(Em-En)
            dr[n, m] -= im*dEm*dSαbar[n, m]/(Em-En)
            dr[n, m] -= im*Em*tmpS[n, m]/(Em-En)
            dr[n, m] -= im*(dEm-dEn)*Dα[n, m]/(Em-En)
            dr[n, m] += tmpA[n, m]
        end
    end
    return dr
end


@doc raw"""
```julia
getvelocity(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Number})
```

Calculate velocity matrix in the α direction.

Velocity matrix is calculated by the following expression
```math
v_{nm}^α = ∂_α ϵ_n δ_{nm} + i (ϵ_n-ϵ_m) A_{nm}^α.
```
Therefore, the velocity is actually ħ*velocity.

v[n, m] is only correct when band m is nondegenerate or completely degenerate.
"""
@memoize k function getvelocity(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    v = zeros(ComplexF64, tm.norbits, tm.norbits)
    # diagonal elements
    dEs = getdEs(tm, α, k)
    for n in 1:tm.norbits
        v[n, n] = dEs[n]
    end
    # off-diagonal elements
    Aα = getA(tm, α, k)
    Es = geteig(tm, k).values
    for m in 1:tm.norbits, n in 1:tm.norbits
        if n != m
            v[n, m] = im*(Es[n]-Es[m])*Aα[n, m]
        end
    end
    return v
end


"""
```julia
getspin(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real})
```

Calculate ⟨n|σα|m⟩ at `k` point.

If `tm` is a TBModel, the function checks whether `tm.isspinful` is true.
"""
@memoize k function getspin(
    tm::AbstractTBModel,
    α::Integer,
    k::AbstractVector{<:Real}
)::Matrix{ComplexF64}
    if tm isa TBModel && !(tm.isspinful === true)
        error("TBModel should be spinful.")
    end
    length(k) == 3 || error("k should be a 3-element vector.")
    α in 1:3 || error("α should be 1, 2 or 3.")
    nspinless = tm.norbits ÷ 2
    V = geteig(tm, k).vectors
    return V' * kron(σs[α], getS(tm, k)[1:nspinless, 1:nspinless]) * V
end


"""
```julia
get_berry_curvature(tm::AbstractTBModel, α::Int64, β::Int64, k::Vector{<:Real})::Vector{Float64}
```

Calculate Berry curvature Ω for `tm` at `k`.

Ω[n] is only correct if band n is nondegenerate or completely degenerate.
"""
@memoize k function get_berry_curvature(tm::AbstractTBModel, α::Int64, β::Int64, k::Vector{<:Real})
    _, V = geteig(tm, k)
    Sbar_α = V' * getdS(tm, getorder(α), k) * V
    Sbar_β = V' * getdS(tm, getorder(β), k) * V
    Abar_α =V'* HopTB.getAw(tm, α, k) *V
    Abar_β =V'* HopTB.getAw(tm, β, k) *V
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    dAw_βα = HopTB.getdAw(tm, β, getorder(α), k)
    dAw_αβ = HopTB.getdAw(tm, α, getorder(β), k)
    Ωbar_αβ = V' * (dAw_βα - dAw_αβ) * V
    return real(diag(Ωbar_αβ - Sbar_α * Abar_β + Sbar_β * Abar_α - im * D_α * D_β + im * D_β * D_α +  D_β * Abar_α - Abar_α * D_β- D_α * Abar_β + Abar_β * D_α ))
end

@memoize k function get_quantum_metric(tm::AbstractTBModel, α::Int64, β::Int64, k::Vector{<:Real})
    _, V = geteig(tm, k)
    Sbar_α = V' * getdS(tm, getorder(α), k) * V
    Sbar_β = V' * getdS(tm, getorder(β), k) * V
    Abar_α =V'* HopTB.getAw(tm, α, k) *V
    Abar_β =V'* HopTB.getAw(tm, β, k) *V
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    dAw_βα = HopTB.getdAw(tm, β, getorder(α), k)
    dAw_αβ = HopTB.getdAw(tm, α, getorder(β), k)
    Gbar_αβ = V' * (dAw_βα + dAw_αβ) * V
    A_ab=0
    return real(-im*diag(Gbar_αβ - Sbar_α * Abar_β - Sbar_β * Abar_α - im * D_α * D_β - im * D_β * D_α -  D_β * Abar_α + Abar_α * D_β - D_α * Abar_β + Abar_β * D_α  ))
end


@memoize k function get_berry_curvature2(tm::AbstractTBModel, α::Int64, β::Int64, k::Vector{<:Real})
    _, V = geteig(tm, k)
    Sbar_α = V' * getdS(tm, getorder(α), k) * V
    Sbar_β = V' * getdS(tm, getorder(β), k) * V
    Abar_α =V'* HopTB.getAw(tm, α, k) *V
    Abar_β =V'* HopTB.getAw(tm, β, k) *V
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    dAw_βα = HopTB.getdAw(tm, β, getorder(α), k)
    dAw_αβ = HopTB.getdAw(tm, α, getorder(β), k)
    Ωbar_αβ = V' * (dAw_βα - dAw_αβ) * V
    return real(diag( D_α' * D_β ))
end

@memoize k function gete(tm::AbstractTBModel, α::Int64, β::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    Ha=HopTB.getHαnm(tm,α,k)
    Hb=HopTB.getHαnm(tm,β,k)
    foo=Ha*D_β-D_α*Hb
    Es = geteig(tm, k).values
    D = zeros(ComplexF64, tm.norbits, tm.norbits)
    for m in 1:tm.norbits, n in 1:tm.norbits
        En = Es[n]; Em = Es[m]
        if abs(En-Em) > DEGEN_THRESH[1]
            D[n, m] = foo[n,m]/(Em-En)
        else
            D[n, m] = 0
        end
    end
    return D
end

@memoize k function gete2(tm::AbstractTBModel, α::Int64, β::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    Hab= HopTB.getdHs(tm,α,β,k)
    Es = geteig(tm, k).values
    D = zeros(ComplexF64, tm.norbits, tm.norbits)
    for m in 1:tm.norbits, n in 1:tm.norbits
        En = Es[n]; Em = Es[m]
        if abs(En-Em) > DEGEN_THRESH[1]
            D[n, m] = Hab[n,m]/(Em-En)
        else
            D[n, m] = 0
        end
    end
    return D
end
"""
```julia
get_berry_curvature(tm::AbstractTBModel, α::Int64, β::Int64, k::Vector{<:Real})::Vector{Float64}
```

Calculate Berry curvature Ω for `tm` at `k`.

Ω[n] is only correct if band n is nondegenerate or completely degenerate.
"""
@memoize k function get_berry_curvature_dipole(tm::AbstractTBModel, α::Int64, β::Int64,γ::Int64, k::Vector{<:Real})
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    eβγ=gete(tm,β,γ,k)
    eγβ=gete(tm,γ,β,k)
    eαγ=gete(tm,α,γ,k)
    eγα=gete(tm,γ,α,k)
    ebc=gete2(tm,β,γ,k)
    eac=gete2(tm,α,γ,k)
    return real(diag(-im*D_α*ebc -im * D_α * eβγ -im * D_α * eγβ+im*D_β*eac+im * D_β * eαγ+im*D_β*eγα))
end

@memoize k function get_berry_curvature_dipole3(tm::AbstractTBModel, α::Int64, β::Int64,γ::Int64, k::Vector{<:Real})
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    eβγ=gete(tm,β,γ,k)
    eγβ=gete(tm,γ,β,k)
    eαγ=gete(tm,α,γ,k)
    eγα=gete(tm,γ,α,k)
    ebc=gete2(tm,β,γ,k)
    eac=gete2(tm,α,γ,k)
    return real(diag(-im*D_α*ebc -im * D_α * eβγ -im * D_α * eγβ+im*D_β*eac+im * D_β * eαγ+im*D_β*eγα-im*eac*D_β -im * eαγ* D_β -im * eγα * D_β+im*ebc*D_α+im * eβγ * D_α+im*eγβ*D_α))
end

@memoize k function get_berry_curvature_dipole2(tm::AbstractTBModel, α::Int64, β::Int64,γ::Int64, k::Vector{<:Real})
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    D_γ = HopTB.getD(tm, γ, k)
    D_αγ = HopTB.getD2(tm,α,γ,k)
    D_βγ = HopTB.getD2(tm,β,γ,k)
    return real(diag(-im*D_α*D_βγ-im*D_αγ*D_β+im*D_γ*D_α*D_β+im*D_α*D_γ*D_β
            +im*D_β*D_αγ+im*D_βγ*D_α-im*D_γ*D_β*D_α-im*D_β*D_γ*D_α))
end

@memoize k function get_berry_curvature_dipole4(tm::AbstractTBModel, α::Int64, β::Int64,γ::Int64, k::Vector{<:Real})
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    D_γ = HopTB.getD(tm, γ, k)
    D_αγ = HopTB.getD2(tm,α,γ,k)
    D_βγ = HopTB.getD2(tm,β,γ,k)
    return real(diag(im*D_α'*D_βγ+im*D_αγ'*D_β
            -im*D_β'*D_αγ-im*D_βγ'*D_α))
end

@memoize k function get_berry_curvature_dipoledft(tm::AbstractTBModel, α::Int64, β::Int64,γ::Int64, k::Vector{<:Real})
     _, V = geteig(tm, k)
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    D_γ = HopTB.getD(tm, γ, k)
    D_αγ = HopTB.getD2(tm,α,γ,k)
    D_βγ = HopTB.getD2(tm,β,γ,k)
    Sbar_α = V' * getdS(tm, getorder(α), k) * V
    Sbar_β = V' * getdS(tm, getorder(β), k) * V
    Sbar_αγ = V' * getdS(tm, getorder(α,γ), k) * V
    Sbar_βγ = V' * getdS(tm, getorder(β,γ), k) * V
    Abar_α =V'* HopTB.getAw(tm, α, k) *V
    Abar_β =V'* HopTB.getAw(tm, β, k) *V
    Abar_αγ =V'* HopTB.getdAw(tm, α,getorder(γ), k) *V
    Abar_βγ =V'* HopTB.getdAw(tm, β,getorder(γ), k) *V
    dAw_βαγ = HopTB.getdAw(tm, β, getorder(α,γ), k)
    dAw_αβγ = HopTB.getdAw(tm, α, getorder(β,γ), k)
    dAw_βα = HopTB.getdAw(tm, β, getorder(α), k)
    dAw_αβ = HopTB.getdAw(tm, α, getorder(β), k)
    Ωbar_αβ = V' * (dAw_βα - dAw_αβ) * V
    Ωbar_αβ_γ = V' * (dAw_βαγ - dAw_αβγ) * V
    return real(diag(Ωbar_αβ_γ 
            - Sbar_αγ * Abar_β + Sbar_βγ * Abar_α  -D_βγ' * Abar_α - Abar_αγ * D_β+ D_αγ' *  Abar_β + Abar_βγ * D_α
            - Sbar_α * Abar_βγ + Sbar_β * Abar_αγ  -D_β' * Abar_αγ - Abar_α * D_βγ+ D_α' *  Abar_βγ + Abar_β * D_αγ
            + D_γ'*Ωbar_αβ+Ωbar_αβ*D_γ
            -D_γ'*Sbar_α*Abar_β + D_γ'*Sbar_β*Abar_α               - D_γ'*Abar_α*D_β                + D_γ'*Abar_β*D_α
            -Sbar_α*Abar_β*D_γ + Sbar_β*Abar_α*D_γ -D_β'*Abar_α*D_γ               + D_α'*Abar_β*D_γ
            -im*D_α*D_βγ-im*D_αγ*D_β+im*D_γ*D_α*D_β+im*D_α*D_γ*D_β
            +im*D_β*D_αγ+im*D_βγ*D_α-im*D_γ*D_β*D_α-im*D_β*D_γ*D_α))
end

@memoize k function get_quantum_metric_dipoledft(tm::AbstractTBModel, α::Int64, β::Int64,γ::Int64, k::Vector{<:Real})
     _, V = geteig(tm, k)
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    D_γ = HopTB.getD(tm, γ, k)
    D_αγ = HopTB.getD2(tm,α,γ,k)
    D_βγ = HopTB.getD2(tm,β,γ,k)
    Sbar_α = V' * getdS(tm, getorder(α), k) * V
    Sbar_β = V' * getdS(tm, getorder(β), k) * V
    Sbar_αγ = V' * getdS(tm, getorder(α,γ), k) * V
    Sbar_βγ = V' * getdS(tm, getorder(β,γ), k) * V
    Abar_α =V'* HopTB.getAw(tm, α, k) *V
    Abar_β =V'* HopTB.getAw(tm, β, k) *V
    Abar_αγ =V'* HopTB.getdAw(tm, α,getorder(γ), k) *V
    Abar_βγ =V'* HopTB.getdAw(tm, β,getorder(γ), k) *V
    dAw_βαγ = HopTB.getdAw(tm, β, getorder(α,γ), k)
    dAw_αβγ = HopTB.getdAw(tm, α, getorder(β,γ), k)
    dAw_βα = HopTB.getdAw(tm, β, getorder(α), k)
    dAw_αβ = HopTB.getdAw(tm, α, getorder(β), k)
    Gbar_αβ = V' * (dAw_βα + dAw_αβ) * V
    Gbar_αβ_γ = V' * (dAw_βαγ + dAw_αβγ) * V
    return real(diag(-im*(Gbar_αβ_γ 
            - Sbar_αγ * Abar_β - Sbar_βγ * Abar_α  -D_βγ' * Abar_α + Abar_αγ * D_β- D_αγ' *  Abar_β + Abar_βγ * D_α
            - Sbar_α * Abar_βγ - Sbar_β * Abar_αγ  -D_β' * Abar_αγ + Abar_α * D_βγ- D_α' *  Abar_βγ + Abar_β * D_αγ
            + D_γ'*Gbar_αβ+Gbar_αβ*D_γ
            -D_γ'*Sbar_α*Abar_β - D_γ'*Sbar_β*Abar_α               - D_γ'*Abar_α*D_β                - D_γ'*Abar_β*D_α
            +Sbar_α*Abar_β*D_γ + Sbar_β*Abar_α*D_γ -D_β'*Abar_α*D_γ               - D_α'*Abar_β*D_γ
            -im*D_α'*D_βγ-im*D_αγ'*D_β
            -im*D_β'*D_αγ-im*D_βγ'*D_α)))
end



@memoize k function get_berry_curvature_dipoledft2(tm::AbstractTBModel, α::Int64, β::Int64,γ::Int64, k::Vector{<:Real})
     _, V = geteig(tm, k)
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    D_γ = HopTB.getD(tm, γ, k)
    D_αγ = HopTB.getD2(tm,α,γ,k)
    D_βγ = HopTB.getD2(tm,β,γ,k)
    Sbar_α = V' * getdS(tm, getorder(α), k) * V
    Sbar_β = V' * getdS(tm, getorder(β), k) * V
    Sbar_αγ = V' * getdS(tm, getorder(α,γ), k) * V
    Sbar_βγ = V' * getdS(tm, getorder(β,γ), k) * V
    Abar_α =V'* HopTB.getAw(tm, α, k) *V
    Abar_β =V'* HopTB.getAw(tm, β, k) *V
    Abar_αγ =V'* HopTB.getdAw(tm, α,getorder(γ), k) *V
    Abar_βγ =V'* HopTB.getdAw(tm, β,getorder(γ), k) *V
    dAw_βαγ = HopTB.getdAw(tm, β, getorder(α,γ), k)
    dAw_αβγ = HopTB.getdAw(tm, α, getorder(β,γ), k)
    dAw_βα = HopTB.getdAw(tm, β, getorder(α), k)
    dAw_αβ = HopTB.getdAw(tm, α, getorder(β), k)
    Ωbar_αβ = V' * (dAw_βα - dAw_αβ) * V
    Ωbar_αβ_γ = V' * (dAw_βαγ - dAw_αβγ) * V
    return real(diag(Ωbar_αβ_γ 
            - Sbar_αγ * Abar_β + Sbar_βγ * Abar_α  +D_βγ * Abar_α - Abar_αγ * D_β- D_αγ *  Abar_β + Abar_βγ * D_α
            - Sbar_α * Abar_βγ + Sbar_β * Abar_αγ  +D_β * Abar_αγ - Abar_α * D_βγ- D_α *  Abar_βγ + Abar_β * D_αγ
            - D_γ*Ωbar_αβ+Ωbar_αβ*D_γ
            +D_γ*Sbar_α*Abar_β - D_γ*Sbar_β*Abar_α               + D_γ*Abar_α*D_β                - D_γ*Abar_β*D_α
            -Sbar_α*Abar_β*D_γ + Sbar_β*Abar_α*D_γ +D_β*Abar_α*D_γ               - D_α*Abar_β*D_γ
            -im*D_α*D_βγ-im*D_αγ*D_β
            +im*D_β*D_αγ+im*D_βγ*D_α))
end

@memoize k function getDHs(
    tm::AbstractTBModel,
    α::Integer,
    β::Integer,
    k::AbstractVector{<:Real}
)::Matrix{ComplexF64}
    Es, _ = geteig(tm, k)
    dHαβbar = getdHs(tm, α, β, k)
    Dβ = getD(tm, β, k)
    dHαbar = getHαnm(tm, α, k)
    dEs = zeros(ComplexF64, tm.norbits, tm.norbits)
    foo1 = -dHαbar * Dβ + Dβ * dHαbar
    for m in 1:tm.norbits, n in 1:tm.norbits
        dEs[n,m] += dHαβbar[n, m]
        dEs[n,m] += foo1[n, m]
    end
    return dEs
end

@memoize k function getDll(tm::AbstractTBModel, α::Int64,β::Int64,γ::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    Es, _ = geteig(tm, k)
    m = 0  # 初始化 m
    n = 0  # 初始化 n
    DHαβγ=HopTB.getDHs(tm,α,β,γ,k)
    DHβγ=HopTB.getDHs(tm,β,γ,k)
    DHαγ=HopTB.getDHs(tm,α,γ,k)
    D_αβ=HopTB.getDl(tm,α,β,k)
    D_αγ=HopTB.getDl(tm,α,γ,k)
    D_βγ=HopTB.getDl(tm,β,γ,k)
    D_β = HopTB.getD(tm, β, k)
    D_α = HopTB.getD(tm, α, k)
    Ha=HopTB.getHαnm(tm,α,k)
    Hb=HopTB.getHαnm(tm,β,k)
    Hc=HopTB.getHαnm(tm,γ,k)
    foo1=DHαγ*D_β+Ha*D_βγ-D_βγ*Ha-D_β*DHαγ
    foo2=DHβγ*D_α+Hb*D_αγ-D_αγ*Hb-D_α*DHβγ
    foo3=Hc*D_αβ-D_αβ*Hc
    Dll = zeros(ComplexF64, tm.norbits, tm.norbits)
    for m in 1:tm.norbits, n in 1:tm.norbits
        En = Es[n]; Em = Es[m]
        if abs(En-Em) > DEGEN_THRESH[1]
            Dll[n, m] += DHαβγ[n,m]/(Em-En)
            Dll[n, m] += foo1[n,m]/(Em-En)
            Dll[n, m] += foo2[n,m]/(Em-En)
            Dll[n, m] += foo3[n,m]/(Em-En)
        else
            Dll[n, m] = 0
        end
    end
    return Dll
end

@memoize k function getD3(tm::AbstractTBModel, α::Int64,β::Int64,γ::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    Es, _ = geteig(tm, k)
    m = 0  # 初始化 m
    n = 0  # 初始化 n
    Es, _ = geteig(tm, k)
    dHαbar = getdHbar(tm, getorder(α), k)
    dHβbar = getdHbar(tm, getorder(β), k)
    dHγbar = getdHbar(tm, getorder(γ), k)
    dSαbar = getdSbar(tm, getorder(α), k)
    dSβbar = getdSbar(tm, getorder(β), k)
    dSγbar = getdSbar(tm, getorder(γ), k)
    dHαβbar = getdHbar(tm, getorder(α, β), k)
    dHβγbar = getdHbar(tm, getorder(β,γ), k)
    dHγαbar = getdHbar(tm, getorder(γ,α), k)
    dSαβbar = getdSbar(tm, getorder(α, β), k)
    dSβγbar = getdSbar(tm, getorder(β,γ), k)
    dSγαbar = getdSbar(tm, getorder(γ,α), k)
    dHαβγbar = getdHbar(tm, getorder(α, β, γ), k)
    dSαβγbar = getdSbar(tm, getorder(α, β, γ), k)
    DHβγ=HopTB.getDHs(tm,β,γ,k)
    DHαγ=HopTB.getDHs(tm,α,γ,k)
    D_αβ=HopTB.getD2(tm,α,β,k)
    D_γα=HopTB.getD2(tm,γ,α,k)
    D_βγ=HopTB.getD2(tm,β,γ,k)
    D_β = HopTB.getD(tm, β, k)
    D_α = HopTB.getD(tm, α, k)
    D_γ = HopTB.getD(tm, γ, k)
    foo3α = getdEs(tm, α, k)
    foo3β = getdEs(tm, β, k)
    foo3γ = getdEs(tm, γ, k)
    foo3αβ = getdEs(tm, α, β, k)
    foo3βγ = getdEs(tm, β, γ, k)
    foo3γα = getdEs(tm, γ, α, k)
    foo4αβ = dSαbar*D_β
    foo4αγ = dSαbar*D_γ
    foo4βα = dSβbar*D_α
    foo4βγ = dSβbar*D_γ
    foo4γα = dSγbar*D_α
    foo4γβ = dSγbar*D_β
    foo1=dHαβbar*D_γ+dHβγbar*D_α+dHγαbar*D_β
    foo1s=dSαβbar*D_γ+dSβγbar*D_α+dSγαbar*D_β
    foo2=dHαbar*D_βγ+dHβbar*D_γα+dHγbar*D_αβ
    foo2s=dSαbar*D_βγ+dSβbar*D_γα+dSγbar*D_αβ
    Dll = zeros(ComplexF64, tm.norbits, tm.norbits)
    Awαβγ = getdAwbar(tm, α, getorder(β,γ), k)
    Awαγ = getdAwbar(tm, α, getorder(γ), k)
    Awα = getAwbar(tm, α, k)
    L=D_βγ*Awα-Awα*D_βγ+D_β*Awαγ-Awαγ*D_β+D_β*Awα*D_γ+D_γ*Awα*D_β
    for m in 1:tm.norbits, n in 1:tm.norbits
        En = Es[n]; Em = Es[m]
        if abs(En-Em) > DEGEN_THRESH[1]
            Dll[n, m] += (dHαβγbar[n,m]- dSαβγbar[n,m]*Em)/(Em-En)
            Dll[n, m] += (foo1[n,m]-foo1s[n,m])/(Em-En)
            Dll[n, m] += (foo2[n,m]-foo2s[n,m])/(Em-En)
            Dll[n, m] -= (foo4αβ[n,m]*foo3γ[m]+foo4αγ[n,m]*foo3β[m]+foo4βα[n,m]*foo3γ[m]+foo4γα[n,m]*foo3β[m]+foo4βγ[n,m]*foo3α[m]+foo4γβ[n,m]*foo3α[m])/(Em-En)
            Dll[n, m] -= (dSαβbar[n,m]*foo3γ[m]+dSβγbar[n,m]*foo3α[m]+dSγαbar[n,m]*foo3β[m])/(Em-En)
            Dll[n, m] -= (D_α[n,m]*foo3βγ[m]+D_β[n,m]*foo3γα[m]+D_γ[n,m]*foo3αβ[m])/(Em-En)
            Dll[n, m] -= (D_αβ[n,m]*foo3γ[m]+D_βγ[n,m]*foo3α[m]+D_γα[n,m]*foo3β[m])/(Em-En)
        else
            Dll[n, m] = 0
        end
    end
    return Dll
end

@memoize k function get_berry_curvature_quadrupole(tm::AbstractTBModel, α::Int64, β::Int64,γ::Int64,δ::Int64, k::Vector{<:Real})
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    D_γ = HopTB.getD(tm, γ, k)
    D_δ = HopTB.getD(tm, δ, k)
    D_αγ = HopTB.getD2(tm,α,γ,k)
    D_αδ = HopTB.getD2(tm,α,δ,k)
    D_βγ = HopTB.getD2(tm,β,γ,k)
    D_βδ = HopTB.getD2(tm,β,δ,k)
    D_αγδ = HopTB.getD3(tm,α,γ,δ,k)
    D_βγδ = HopTB.getD3(tm,β,γ,δ,k)
    return real(diag(im*D_α'*D_βγδ+im*D_αδ'*D_βγ
            +im*D_αγδ'*D_β+im*D_αγ'*D_βδ
            -im*D_βγδ'*D_α-im*D_β'*D_αγδ
            -im*D_βγ'*D_αδ-im*D_βδ'*D_αγ))
end

@memoize k function get_quatum_metric_quadrupole(tm::AbstractTBModel, α::Int64, β::Int64,γ::Int64,δ::Int64, k::Vector{<:Real})
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    D_γ = HopTB.getD(tm, γ, k)
    D_δ = HopTB.getD(tm, δ, k)
    D_αγ = HopTB.getD2(tm,α,γ,k)
    D_αδ = HopTB.getD2(tm,α,δ,k)
    D_βγ = HopTB.getD2(tm,β,γ,k)
    D_βδ = HopTB.getD2(tm,β,δ,k)
    D_αγδ = HopTB.getD3(tm,α,γ,δ,k)
    D_βγδ = HopTB.getD3(tm,β,γ,δ,k)
    return real(-im*diag(-im*D_α'*D_βγδ-im*D_αδ'*D_βγ
            -im*D_αγδ'*D_β-im*D_αγ'*D_βδ
            -im*D_βγδ'*D_α-im*D_β'*D_αγδ
            -im*D_βγ'*D_αδ-im*D_βδ'*D_αγ))
end


@memoize k function get_berry_curvature_quadrupolef(tm::AbstractTBModel, α::Int64, β::Int64,γ::Int64,δ::Int64, k::Vector{<:Real})
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    D_γ = HopTB.getD(tm, γ, k)
    D_δ = HopTB.getD(tm, δ, k)
    D_αγ = HopTB.getD2(tm,α,γ,k)
    D_αδ = HopTB.getD2(tm,α,δ,k)
    D_βγ = HopTB.getD2(tm,β,γ,k)
    D_βδ = HopTB.getD2(tm,β,δ,k)
    D_γδ = HopTB.getD2(tm,γ,δ,k)
    D_αγδ = HopTB.getD3(tm,α,γ,δ,k)
    D_βγδ = HopTB.getD3(tm,β,γ,δ,k)
    return real(diag(-im*D_α*D_βγδ-im*D_αδ*D_βγ
            -im*D_δ*D_α*D_γ*D_β-im*D_α*D_δ*D_γ*D_β-im*D_α*D_γ*D_δ*D_β
            +im*D_αδ*D_γ*D_β+im*D_α*D_γδ*D_β+im*D_α*D_γ*D_βδ+im*D_δ*D_α*D_βγ+im*D_α*D_δ*D_βγ
            -im*D_αγδ*D_β-im*D_αγ*D_βδ
            -im*D_γ*D_α*D_δ*D_β-im*D_γ*D_δ*D_α*D_β-im*D_δ*D_γ*D_α*D_β
            +im*D_γδ*D_α*D_β+im*D_γ*D_αδ*D_β+im*D_γ*D_α*D_βδ+im*D_αγ*D_δ*D_β+im*D_δ*D_αγ*D_β
            +im*D_βγδ*D_α+im*D_β*D_αγδ
            +im*D_δ*D_β*D_γ*D_α+im*D_β*D_δ*D_γ*D_α+im*D_β*D_γ*D_δ*D_α
            -im*D_βδ*D_γ*D_α-im*D_β*D_γδ*D_α-im*D_β*D_γ*D_αδ-im*D_δ*D_β*D_αγ-im*D_β*D_δ*D_αγ
            +im*D_βγ*D_αδ+im*D_βδ*D_αγ
            +im*D_γ*D_β*D_δ*D_α+im*D_δ*D_γ*D_β*D_α+im*D_γ*D_δ*D_β*D_α
            -im*D_γδ*D_β*D_α-im*D_γ*D_βδ*D_α-im*D_γ*D_β*D_αδ-im*D_βγ*D_δ*D_α-im*D_δ*D_βγ*D_α))
end

@memoize k function get_berry_curvature_quadrupole2(tm::AbstractTBModel, α::Int64, β::Int64,γ::Int64,δ::Int64, k::Vector{<:Real})
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    D_αγ = HopTB.getD2(tm,α,γ,k)
    D_αδ = HopTB.getD2(tm,α,δ,k)
    D_βγ = HopTB.getD2(tm,β,γ,k)
    D_βδ = HopTB.getD2(tm,β,δ,k)
    D_αγδ = HopTB.getD3(tm,α,γ,δ,k)
    D_βγδ = HopTB.getD3(tm,β,γ,δ,k)
    return real(diag(-im*D_α*D_βγδ-im*D_αδ*D_βγ -im*D_αγδ*D_β-im*D_αγ*D_βδ))
end

@memoize k function get_berry_curvature_quadrupole3(tm::AbstractTBModel, α::Int64, β::Int64,γ::Int64,δ::Int64, k::Vector{<:Real})
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    D_αγ = HopTB.getD2(tm,α,γ,k)
    D_αδ = HopTB.getD2(tm,α,δ,k)
    D_βγ = HopTB.getD2(tm,β,γ,k)
    D_βδ = HopTB.getD2(tm,β,δ,k)
    D_αγδ = HopTB.getD3(tm,α,γ,δ,k)
    D_βγδ = HopTB.getD3(tm,β,γ,δ,k)
    return real(diag(-im*D_α*D_βγδ+im * D_β * D_αγδ))
end

@memoize k function get_berry_curvature_quadrupoledft(tm::AbstractTBModel, α::Int64, β::Int64,γ::Int64,δ::Int64, k::Vector{<:Real})
    _, V = geteig(tm, k)
    Sbar_α = V' * getdS(tm, getorder(α), k) * V
    Sbar_β = V' * getdS(tm, getorder(β), k) * V
    Sbar_αγ = V' * getdS(tm, getorder(α,γ), k) * V
    Sbar_βγ = V' * getdS(tm, getorder(β,γ), k) * V
    Sbar_αδ = V' * getdS(tm, getorder(α,δ), k) * V
    Sbar_βδ = V' * getdS(tm, getorder(β,δ), k) * V
    Sbar_αγδ = V' * getdS(tm, getorder(α,γ,δ), k) * V
    Sbar_βγδ = V' * getdS(tm, getorder(β,γ,δ), k) * V
    Abar_α =V'* HopTB.getAw(tm, α, k) *V
    Abar_β =V'* HopTB.getAw(tm, β, k) *V
    Abar_αγ =V'* HopTB.getdAw(tm, α,getorder(γ), k) *V
    Abar_βγ =V'* HopTB.getdAw(tm, β,getorder(γ), k) *V
    Abar_αδ =V'* HopTB.getdAw(tm, α,getorder(δ), k) *V
    Abar_βδ =V'* HopTB.getdAw(tm, β,getorder(δ), k) *V
    Abar_αγδ =V'* HopTB.getdAw(tm, α,getorder(γ,δ), k) *V
    Abar_βγδ =V'* HopTB.getdAw(tm, β,getorder(γ,δ), k) *V
    dAw_βαγδ = HopTB.getdAw(tm, β, getorder(α,γ,δ), k)
    dAw_αβγδ = HopTB.getdAw(tm, α, getorder(β,γ,δ), k)
    dAw_βαδ = HopTB.getdAw(tm, β, getorder(α,δ), k)
    dAw_αβδ = HopTB.getdAw(tm, α, getorder(β,δ), k)
    dAw_βαγ = HopTB.getdAw(tm, β, getorder(α,γ), k)
    dAw_αβγ = HopTB.getdAw(tm, α, getorder(β,γ), k)
    dAw_βα = HopTB.getdAw(tm, β, getorder(α), k)
    dAw_αβ = HopTB.getdAw(tm, α, getorder(β), k)
    Ωbar_αβ = V' * (dAw_βα - dAw_αβ) * V
    Ωbar_αβδ = V' * (dAw_βαδ - dAw_αβδ) * V
    Ωbar_αβ_γ = V' * (dAw_βαγ - dAw_αβγ) * V
    Ωbar_αβ_γδ = V' * (dAw_βαγδ - dAw_αβγδ) * V
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    D_γ = HopTB.getD(tm, γ, k)
    D_δ = HopTB.getD(tm, δ, k)
    D_αγ = HopTB.getD2(tm,α,γ,k)
    D_αδ = HopTB.getD2(tm,α,δ,k)
    D_βγ = HopTB.getD2(tm,β,γ,k)
    D_βδ = HopTB.getD2(tm,β,δ,k)
    D_γδ = HopTB.getD2(tm,γ,δ,k)
    D_αγδ = HopTB.getD3(tm,α,γ,δ,k)
    D_βγδ = HopTB.getD3(tm,β,γ,δ,k)
    return real(diag(Ωbar_αβ_γδ 
            - Sbar_αγδ * Abar_β + Sbar_βγδ * Abar_α  +D_βγδ * Abar_α - Abar_αγδ * D_β- D_αγδ *  Abar_β + Abar_βγδ * D_α
                - Sbar_αγ * Abar_βδ + Sbar_βγ * Abar_αδ  +D_βγ * Abar_αδ - Abar_αγ * D_βδ- D_αγ *  Abar_βδ + Abar_βγ * D_αδ
            - Sbar_αδ * Abar_βγ + Sbar_βδ * Abar_αγ  +D_βδ * Abar_αγ - Abar_αδ * D_βγ- D_αδ *  Abar_βγ + Abar_βδ * D_αγ
                - Sbar_α * Abar_βγδ + Sbar_β * Abar_αγδ  +D_β * Abar_αγδ - Abar_α * D_βγδ- D_α *  Abar_βγδ + Abar_β * D_αγδ
            - D_γδ*Ωbar_αβ+Ωbar_αβδ*D_γ
                - D_γ*Ωbar_αβδ+Ωbar_αβ*D_γδ
            +D_γδ*Sbar_α*Abar_β - D_γδ*Sbar_β*Abar_α +              + D_γδ*Abar_α*D_β                - D_γδ*Abar_β*D_α
                +D_γ*Sbar_αδ*Abar_β - D_γ*Sbar_βδ*Abar_α +              + D_γ*Abar_αδ*D_β                - D_γ*Abar_βδ*D_α
                    +D_γ*Sbar_α*Abar_βδ - D_γ*Sbar_β*Abar_αδ +              + D_γ*Abar_α*D_βδ                - D_γ*Abar_β*D_αδ
            -Sbar_αδ*Abar_β*D_γ + Sbar_βδ*Abar_α*D_γ +D_βδ*Abar_α*D_γ               - D_αδ*Abar_β*D_γ
                -Sbar_α*Abar_βδ*D_γ + Sbar_β*Abar_αδ*D_γ +D_β*Abar_αδ*D_γ               - D_α*Abar_βδ*D_γ
                    -Sbar_α*Abar_β*D_γδ + Sbar_β*Abar_α*D_γδ +D_β*Abar_α*D_γδ               - D_α*Abar_β*D_γδ
                                                                  - Abar_αδ*D_γ*D_β                + Abar_βδ*D_γ*D_α
                                                   -D_βδ*D_γ*Abar_α                                + D_αδ*D_γ*Abar_β
                                                                                  - Abar_α*D_γδ*D_β                + Abar_β*D_γδ*D_α
                                                       -D_β*D_γδ*Abar_α                                + D_α*D_γδ*Abar_β
                                                                                        - Abar_α*D_γ*D_βδ                + Abar_β*D_γ*D_αδ
                                                            -D_β*D_γ*Abar_αδ                                + D_α*D_γ*Abar_βδ
            - D_δ*(Ωbar_αβ_γ 
            - Sbar_αγ * Abar_β + Sbar_βγ * Abar_α                 - Abar_αγ * D_β                 + Abar_βγ * D_α
            - Sbar_α * Abar_βγ + Sbar_β * Abar_αγ                 - Abar_α * D_βγ                 + Abar_β * D_αγ
                               +Ωbar_αβ*D_γ
            
            -Sbar_α*Abar_β*D_γ + Sbar_β*Abar_α*D_γ 
                                                                  - Abar_α*D_γ*D_β                + Abar_β*D_γ*D_α
                                                                                                                     )
                                                   -D_βγ*D_δ*Abar_α              + D_αγ*D_δ*Abar_β
                                                   -D_β*D_δ*Abar_αγ              + D_α*D_δ*Abar_βγ
            + D_γ*D_δ*Ωbar_αβ
            - D_γ*D_δ*Sbar_α*Abar_β + D_γ*D_δ*Sbar_β*Abar_α       - D_γ*D_δ*Abar_α*D_β            + D_γ*D_δ*Abar_β*D_α
                                                   -D_β*D_δ*Abar_α*D_γ           + D_α*D_δ*Abar_β*D_γ
                                                   +D_β*D_γ*D_δ*Abar_α                            - D_α*D_γ*D_δ*Abar_β
            + (Ωbar_αβ_γ 
            - Sbar_αγ * Abar_β + Sbar_βγ * Abar_α  +D_βγ * Abar_α                - D_αγ *  Abar_β 
            - Sbar_α * Abar_βγ + Sbar_β * Abar_αγ  +D_β * Abar_αγ                - D_α *  Abar_βγ
            - D_γ*Ωbar_αβ
            +D_γ*Sbar_α*Abar_β - D_γ*Sbar_β*Abar_α                               
            
                                                                  
                                                   -D_β*D_γ*Abar_α                                + D_α*D_γ*Abar_β)*D_δ
                                                                  - Abar_αγ*D_δ*D_β               + Abar_βγ*D_δ*D_α
                                                                  - Abar_α*D_δ*D_βγ               + Abar_β*D_δ*D_αγ
                         +Ωbar_αβ*D_δ*D_γ
                                                                  + D_γ*Abar_α*D_δ*D_β            - D_γ*Abar_β*D_δ*D_α
            -Sbar_α*Abar_β*D_δ*D_γ + Sbar_β*Abar_α*D_δ*D_γ +D_β*Abar_α*D_δ*D_γ               - D_α*Abar_β*D_δ*D_γ
                                                                  - Abar_α*D_δ*D_γ*D_β            + Abar_β*D_δ*D_γ*D_α
            -im*D_α*D_βγδ-im*D_αδ*D_βγ
            -im*D_αγδ*D_β-im*D_αγ*D_βδ
            +im*D_βγδ*D_α+im*D_β*D_αγδ
            +im*D_βγ*D_αδ+im*D_βδ*D_αγ))
end

@memoize k function get_berry_curvature_quadrupoledft2(tm::AbstractTBModel, α::Int64, β::Int64,γ::Int64,δ::Int64, k::Vector{<:Real})
    _, V = geteig(tm, k)
    Sbar_α = V' * getdS(tm, getorder(α), k) * V
    Sbar_β = V' * getdS(tm, getorder(β), k) * V
    Sbar_αγ = V' * getdS(tm, getorder(α,γ), k) * V
    Sbar_βγ = V' * getdS(tm, getorder(β,γ), k) * V
    Sbar_αδ = V' * getdS(tm, getorder(α,δ), k) * V
    Sbar_βδ = V' * getdS(tm, getorder(β,δ), k) * V
    Sbar_αγδ = V' * getdS(tm, getorder(α,γ,δ), k) * V
    Sbar_βγδ = V' * getdS(tm, getorder(β,γ,δ), k) * V
    Abar_α =V'* HopTB.getAw(tm, α, k) *V
    Abar_β =V'* HopTB.getAw(tm, β, k) *V
    Abar_αγ =V'* HopTB.getdAw(tm, α,getorder(γ), k) *V
    Abar_βγ =V'* HopTB.getdAw(tm, β,getorder(γ), k) *V
    Abar_αδ =V'* HopTB.getdAw(tm, α,getorder(δ), k) *V
    Abar_βδ =V'* HopTB.getdAw(tm, β,getorder(δ), k) *V
    Abar_αγδ =V'* HopTB.getdAw(tm, α,getorder(γ,δ), k) *V
    Abar_βγδ =V'* HopTB.getdAw(tm, β,getorder(γ,δ), k) *V
    dAw_βαγδ = HopTB.getdAw(tm, β, getorder(α,γ,δ), k)
    dAw_αβγδ = HopTB.getdAw(tm, α, getorder(β,γ,δ), k)
    dAw_βαδ = HopTB.getdAw(tm, β, getorder(α,δ), k)
    dAw_αβδ = HopTB.getdAw(tm, α, getorder(β,δ), k)
    dAw_βαγ = HopTB.getdAw(tm, β, getorder(α,γ), k)
    dAw_αβγ = HopTB.getdAw(tm, α, getorder(β,γ), k)
    dAw_βα = HopTB.getdAw(tm, β, getorder(α), k)
    dAw_αβ = HopTB.getdAw(tm, α, getorder(β), k)
    Ωbar_αβ = V' * (dAw_βα - dAw_αβ) * V
    Ωbar_αβδ = V' * (dAw_βαδ - dAw_αβδ) * V
    Ωbar_αβ_γ = V' * (dAw_βαγ - dAw_αβγ) * V
    Ωbar_αβ_γδ = V' * (dAw_βαγδ - dAw_αβγδ) * V
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    D_γ = HopTB.getD(tm, γ, k)
    D_δ = HopTB.getD(tm, δ, k)
    D_αγ = HopTB.getD2(tm,α,γ,k)
    D_αδ = HopTB.getD2(tm,α,δ,k)
    D_βγ = HopTB.getD2(tm,β,γ,k)
    D_βδ = HopTB.getD2(tm,β,δ,k)
    D_γδ = HopTB.getD2(tm,γ,δ,k)
    D_αγδ = HopTB.getD3(tm,α,γ,δ,k)
    D_βγδ = HopTB.getD3(tm,β,γ,δ,k)
    return real(diag(Ωbar_αβ_γδ 
            - Sbar_αγδ * Abar_β + Sbar_βγδ * Abar_α  +D_βγδ * Abar_α - Abar_αγδ * D_β- D_αγδ *  Abar_β + Abar_βγδ * D_α
                - Sbar_αγ * Abar_βδ + Sbar_βγ * Abar_αδ  +D_βγ * Abar_αδ - Abar_αγ * D_βδ- D_αγ *  Abar_βδ + Abar_βγ * D_αδ
            - Sbar_αδ * Abar_βγ + Sbar_βδ * Abar_αγ  +D_βδ * Abar_αγ - Abar_αδ * D_βγ- D_αδ *  Abar_βγ + Abar_βδ * D_αγ
                - Sbar_α * Abar_βγδ + Sbar_β * Abar_αγδ  +D_β * Abar_αγδ - Abar_α * D_βγδ- D_α *  Abar_βγδ + Abar_β * D_αγδ
            - D_γδ*Ωbar_αβ+Ωbar_αβδ*D_γ
                - D_γ*Ωbar_αβδ+Ωbar_αβ*D_γδ
            +D_γδ*Sbar_α*Abar_β - D_γδ*Sbar_β*Abar_α               + D_γδ*Abar_α*D_β                - D_γδ*Abar_β*D_α
                +D_γ*Sbar_αδ*Abar_β - D_γ*Sbar_βδ*Abar_α               + D_γ*Abar_αδ*D_β                - D_γ*Abar_βδ*D_α
                    +D_γ*Sbar_α*Abar_βδ - D_γ*Sbar_β*Abar_αδ               + D_γ*Abar_α*D_βδ                - D_γ*Abar_β*D_αδ
            -Sbar_αδ*Abar_β*D_γ + Sbar_βδ*Abar_α*D_γ +D_βδ*Abar_α*D_γ               - D_αδ*Abar_β*D_γ
                -Sbar_α*Abar_βδ*D_γ + Sbar_β*Abar_αδ*D_γ +D_β*Abar_αδ*D_γ               - D_α*Abar_βδ*D_γ
                    -Sbar_α*Abar_β*D_γδ + Sbar_β*Abar_α*D_γδ +D_β*Abar_α*D_γδ               - D_α*Abar_β*D_γδ
            - D_δ*(Ωbar_αβ_γ 
            - Sbar_αγ * Abar_β + Sbar_βγ * Abar_α                 - Abar_αγ * D_β                 + Abar_βγ * D_α
            - Sbar_α * Abar_βγ + Sbar_β * Abar_αγ                 - Abar_α * D_βγ                 + Abar_β * D_αγ
                               +Ωbar_αβ*D_γ
            -Sbar_α*Abar_β*D_γ + Sbar_β*Abar_α*D_γ )
            + (Ωbar_αβ_γ 
            - Sbar_αγ * Abar_β + Sbar_βγ * Abar_α  +D_βγ * Abar_α                - D_αγ *  Abar_β 
            - Sbar_α * Abar_βγ + Sbar_β * Abar_αγ  +D_β * Abar_αγ                - D_α *  Abar_βγ
            - D_γ*Ωbar_αβ
            +D_γ*Sbar_α*Abar_β - D_γ*Sbar_β*Abar_α                               )*D_δ
            -im*D_α*D_βγδ-im*D_αδ*D_βγ
            -im*D_αγδ*D_β-im*D_αγ*D_βδ
            +im*D_βγδ*D_α+im*D_β*D_αγδ
            +im*D_βγ*D_αδ+im*D_βδ*D_αγ))
end


@memoize k function get_berry_curvature_quadrupoledft3(tm::AbstractTBModel, α::Int64, β::Int64,γ::Int64,δ::Int64, k::Vector{<:Real})
    _, V = geteig(tm, k)
    Sbar_α = V' * getdS(tm, getorder(α), k) * V
    Sbar_β = V' * getdS(tm, getorder(β), k) * V
    Sbar_αγ = V' * getdS(tm, getorder(α,γ), k) * V
    Sbar_βγ = V' * getdS(tm, getorder(β,γ), k) * V
    Sbar_αδ = V' * getdS(tm, getorder(α,δ), k) * V
    Sbar_βδ = V' * getdS(tm, getorder(β,δ), k) * V
    Sbar_αγδ = V' * getdS(tm, getorder(α,γ,δ), k) * V
    Sbar_βγδ = V' * getdS(tm, getorder(β,γ,δ), k) * V
    Abar_α =V'* HopTB.getAw(tm, α, k) *V
    Abar_β =V'* HopTB.getAw(tm, β, k) *V
    Abar_αγ =V'* HopTB.getdAw(tm, α,getorder(γ), k) *V
    Abar_βγ =V'* HopTB.getdAw(tm, β,getorder(γ), k) *V
    Abar_αδ =V'* HopTB.getdAw(tm, α,getorder(δ), k) *V
    Abar_βδ =V'* HopTB.getdAw(tm, β,getorder(δ), k) *V
    Abar_αγδ =V'* HopTB.getdAw(tm, α,getorder(γ,δ), k) *V
    Abar_βγδ =V'* HopTB.getdAw(tm, β,getorder(γ,δ), k) *V
    dAw_βαγδ = HopTB.getdAw(tm, β, getorder(α,γ,δ), k)
    dAw_αβγδ = HopTB.getdAw(tm, α, getorder(β,γ,δ), k)
    dAw_βαδ = HopTB.getdAw(tm, β, getorder(α,δ), k)
    dAw_αβδ = HopTB.getdAw(tm, α, getorder(β,δ), k)
    dAw_βαγ = HopTB.getdAw(tm, β, getorder(α,γ), k)
    dAw_αβγ = HopTB.getdAw(tm, α, getorder(β,γ), k)
    dAw_βα = HopTB.getdAw(tm, β, getorder(α), k)
    dAw_αβ = HopTB.getdAw(tm, α, getorder(β), k)
    Ωbar_αβ = V' * (dAw_βα - dAw_αβ) * V
    Ωbar_αβδ = V' * (dAw_βαδ - dAw_αβδ) * V
    Ωbar_αβ_γ = V' * (dAw_βαγ - dAw_αβγ) * V
    Ωbar_αβ_γδ = V' * (dAw_βαγδ - dAw_αβγδ) * V
    D_α = HopTB.getD(tm, α, k)
    D_β = HopTB.getD(tm, β, k)
    D_γ = HopTB.getD(tm, γ, k)
    D_δ = HopTB.getD(tm, δ, k)
    D_αγ = HopTB.getDl(tm,α,γ,k)
    D_αδ = HopTB.getDl(tm,α,δ,k)
    D_βγ = HopTB.getDl(tm,β,γ,k)
    D_βδ = HopTB.getDl(tm,β,δ,k)
    D_γδ = HopTB.getDl(tm,γ,δ,k)
    D_αγδ = HopTB.getDll(tm,α,γ,δ,k)
    D_βγδ = HopTB.getDll(tm,β,γ,δ,k)
    return real(diag(Ωbar_αβ_γδ 
            - Sbar_αγδ * Abar_β + Sbar_βγδ * Abar_α  +D_βγδ * Abar_α - Abar_αγδ * D_β- D_αγδ *  Abar_β + Abar_βγδ * D_α
                - Sbar_αγ * Abar_βδ + Sbar_βγ * Abar_αδ  +D_βγ * Abar_αδ - Abar_αγ * D_βδ- D_αγ *  Abar_βδ + Abar_βγ * D_αδ
            - Sbar_αδ * Abar_βγ + Sbar_βδ * Abar_αγ  +D_βδ * Abar_αγ - Abar_αδ * D_βγ- D_αδ *  Abar_βγ + Abar_βδ * D_αγ
                - Sbar_α * Abar_βγδ + Sbar_β * Abar_αγδ  +D_β * Abar_αγδ - Abar_α * D_βγδ- D_α *  Abar_βγδ + Abar_β * D_αγδ
            - D_γδ*Ωbar_αβ+Ωbar_αβδ*D_γ
                - D_γ*Ωbar_αβδ+Ωbar_αβ*D_γδ
            +D_γδ*Sbar_α*Abar_β - D_γδ*Sbar_β*Abar_α               + D_γδ*Abar_α*D_β                - D_γδ*Abar_β*D_α
                +D_γ*Sbar_αδ*Abar_β - D_γ*Sbar_βδ*Abar_α               + D_γ*Abar_αδ*D_β                - D_γ*Abar_βδ*D_α
                    +D_γ*Sbar_α*Abar_βδ - D_γ*Sbar_β*Abar_αδ               + D_γ*Abar_α*D_βδ                - D_γ*Abar_β*D_αδ
            -Sbar_αδ*Abar_β*D_γ + Sbar_βδ*Abar_α*D_γ +D_βδ*Abar_α*D_γ               - D_αδ*Abar_β*D_γ
                -Sbar_α*Abar_βδ*D_γ + Sbar_β*Abar_αδ*D_γ +D_β*Abar_αδ*D_γ               - D_α*Abar_βδ*D_γ
                    -Sbar_α*Abar_β*D_γδ + Sbar_β*Abar_α*D_γδ +D_β*Abar_α*D_γδ               - D_α*Abar_β*D_γδ
            - D_δ*(Ωbar_αβ_γ 
            - Sbar_αγ * Abar_β + Sbar_βγ * Abar_α                 - Abar_αγ * D_β                 + Abar_βγ * D_α
            - Sbar_α * Abar_βγ + Sbar_β * Abar_αγ                 - Abar_α * D_βγ                 + Abar_β * D_αγ
                               +Ωbar_αβ*D_γ
            -Sbar_α*Abar_β*D_γ + Sbar_β*Abar_α*D_γ )
            + (Ωbar_αβ_γ 
            - Sbar_αγ * Abar_β + Sbar_βγ * Abar_α  +D_βγ * Abar_α                - D_αγ *  Abar_β 
            - Sbar_α * Abar_βγ + Sbar_β * Abar_αγ  +D_β * Abar_αγ                - D_α *  Abar_βγ
            - D_γ*Ωbar_αβ
            +D_γ*Sbar_α*Abar_β - D_γ*Sbar_β*Abar_α                               )*D_δ
            -im*D_α*D_βγδ-im*D_αδ*D_βγ
            -im*D_αγδ*D_β-im*D_αγ*D_βδ
            +im*D_βγδ*D_α+im*D_β*D_αγδ
            +im*D_βγ*D_αδ+im*D_βδ*D_αγ))
end

