__precompile__()
module Fluids

import Base: dot ,sqrt , +, -, *, /, ^, mod, mean, std, size, IndexStyle, getindex, setindex!, eltype, transpose

import Derivatives: dx, dy, dz

using ReadGlobal, MappedArrays, StaticArrays

export Grid, Scalar, ConstScalar, FVector, Tensor, SymTensor, dx, dy, dz, ∇, ∇² , AntiSymTensor, ⊗

struct Grid
  xl::Float64
  yl::Float64
  zl::Float64
end

# =============================== Scalars ==========================================

abstract type AbstractScalar{T,N} <: AbstractArray{T,N} end

^(s::AbstractScalar,n::T) where T<:Union{AbstractScalar,Number} = Scalar(s.grid, s .^ n)
^(s::AbstractScalar,n::Integer) = Scalar(s.grid, s .^ n)
^(n::Number,s::AbstractScalar) = Scalar(s.grid, n .^ s)
+(s::AbstractScalar,n::T) where T<:Union{Number,AbstractScalar} = Scalar(s.grid, s .+ n)
+(s::AbstractScalar,n::T) where T<:Union{Number} = Scalar(s.grid, s .+ n)
+(n::Number,s::AbstractScalar) = Scalar(s.grid, s .+ n)
-(s::AbstractScalar,n::T) where T<:Union{Number,AbstractScalar} = Scalar(s.grid, s .- n)
-(s::AbstractScalar,n::T) where T<:Union{Number} = Scalar(s.grid, s .- n)
-(n::Number,s::AbstractScalar) = Scalar(s.grid, n .- s)
/(s::AbstractScalar,n::T) where T<:Union{Number,AbstractScalar} = Scalar(s.grid, s ./ n)
/(s::AbstractScalar,n::T) where T<:Union{Number} = Scalar(s.grid, s ./ n)
/(n::Number,s::AbstractScalar) = Scalar(s.grid, n ./ s)
*(s::AbstractScalar,n::T) where T<:Union{Number,AbstractScalar} = Scalar(s.grid, s .* n)
*(s::AbstractScalar,n::T) where T<:Union{Number} = Scalar(s.grid, s .* n)
*(n::Number,s::AbstractScalar) = Scalar(s.grid, n .* s)
sqrt(s::AbstractScalar) = Scalar(s.grid, sqrt.(s))
-(s::AbstractScalar) = Scalar(s.grid, -(s))
mean(s::AbstractScalar) = mean(s.data)
std(s::AbstractScalar) = std(s.data)

std(n::T) where T<:Number = zero(T)

struct Scalar{T<:Number,N} <: AbstractScalar{T,N}
  grid::Grid
  data::Array{T,N}
end

Scalar(filename::String) = begin
  nx,ny,nz,xl,yl,zl = getdimsize()
  Scalar(Grid(xl,yl,zl),readfield(filename,nx,ny,nz))
end

size(S::Scalar) = size(S.data)
IndexStyle(::Type{T}) where {T<:Scalar} = Base.IndexLinear()
getindex(S::Scalar,I...) = S.data[I...]
setindex!(S::Scalar,v,I) =  setindex!(S.c,v,I)
eltype(S::Scalar{T,N}) where {T,N} = T

dx(field::Scalar,n::Integer=1) = Scalar(field.grid,dx(field.data,field.grid.xl,n))
dy(field::Scalar,n::Integer=1) = Scalar(field.grid,dy(field.data,field.grid.yl,n))
dz(field::Scalar,n::Integer=1) = Scalar(field.grid,dz(field.data,field.grid.zl,n))

struct ConstScalar{T<:Number,N} <: AbstractScalar{T,N}
  grid::Grid
  size::NTuple{N,Int}
  data::T
end

size(S::ConstScalar) = S.size
IndexStyle(::Type{T}) where {T<:ConstScalar} = Base.IndexCartesian()
getindex(S::ConstScalar,I...) = S.data
eltype(S::ConstScalar{T,N}) where {T,N} = T

^(s::ConstScalar,n::Number) = ConstScalar(s.grid,s.size, s.data^n)
^(n::Number,s::ConstScalar) = ConstScalar(s.grid,s.size, n^s.data)
+(s::ConstScalar,n::Number) = ConstScalar(s.grid,s.size, s.data+n)
+(n::Number,s::ConstScalar) = ConstScalar(s.grid, s.data + n)
-(s::ConstScalar,n::Number) = ConstScalar(s.grid,s.size, s.data - n)
-(n::Number,s::ConstScalar) = ConstScalar(s.grid,s.size, n - s.data)
/(s::ConstScalar,n::Number) = ConstScalar(s.grid,s.size, s.data / n)
/(n::Number,s::ConstScalar) = ConstScalar(s.grid,s.size, n / s.data)
*(s::ConstScalar,n::Number) = ConstScalar(s.grid, s.data * n)
*(n::Number,s::ConstScalar) = ConstScalar(s.grid,s.size, n * s.data)
sqrt(s::ConstScalar) = ConstScalar(s.grid,s.size, sqrt(s.data))
-(s::ConstScalar) = ConstScalar(s.grid,s.size, -s.data)

# =============================== Vectors ==========================================

struct FVector{T<:AbstractFloat,N}
  grid::Grid
  x::Array{T,N}
  y::Array{T,N}
  z::Array{T,N}
end

FVector(file1::String,file2::String,file3::String) = begin
  nx,ny,nz,xl,yl,zl = getdimsize()
  FVector(Grid(xl,yl,zl),readfield(file1,nx,ny,nz),readfield(file2,nx,ny,nz),readfield(file3,nx,ny,nz))
end

getindex(u::FVector,I...) = SVector{3}(u.x[I...],u.y[I...],u.z[I...])

dx(field::FVector,n::Integer=1) = FVector(field.grid,
                          dx(field.x,field.grid.xl,n),
                          dx(field.y,field.grid.xl,n),
                          dx(field.z,field.grid.xl,n))
dy(field::FVector,n::Integer=1) = FVector(field.grid,
                          dy(field.x,field.grid.yl,n),
                          dy(field.y,field.grid.yl,n),
                          dy(field.z,field.grid.yl,n))
dz(field::FVector,n::Integer=1) = FVector(field.grid,
                          dz(field.x,field.grid.zl,n),
                          dz(field.y,field.grid.zl,n),
                          dz(field.z,field.grid.zl,n))

+(v::FVector,u::FVector) = FVector(v.grid,v.x .+ u.x, v.y .+ u.y, v.z .+ u.z)
-(v::FVector,u::FVector) = FVector(v.grid,v.x .- u.x, v.y .- u.y, v.z .- u.z)
-(v::FVector) = FVector(v.grid,-v.x, -v.y, -v.z)
⊗(v::FVector,u::FVector) = Tensor(v.grid,
v.x .* u.x, v.x .* u.y, v.x .* u.z,
v.y .* u.x, v.y .* u.y, v.y .* u.z,
v.z .* u.x, v.z .* u.y, v.z .* u.z)

mean(v::FVector) = SVector{3}(mean(v.x),mean(v.y),mean(v.z))
std(v::FVector;mean=nothing) = mean === nothing ? SVector{3}(std(v.x),std(v.y),std(v.z)) : SVector{3}(std(v.x,mean=mean[1]),std(v.y,mean=mean[2]),std(v.z,mean=mean[3]))

# =============================== Tensors ==========================================

abstract type AbstractTensor{T,N} end

getindex(S::AbstractTensor,I...) = @SMatrix [S.xx[I...] S.xy[I...] S.xz[I...];
                                          S.yx[I...] S.yy[I...] S.yz[I...];
                                          S.zx[I...] S.zy[I...] S.zz[I...]]


-(A::AbstractTensor) = Tensor(A.grid,
-A.xx , -A.xy , -A.xz ,
-A.yx , -A.yy , -A.yz ,
-A.zx , -A.zy , -A.zz )


+(A::AbstractTensor,B::AbstractTensor) = Tensor(A.grid,
A.xx .+ B.xx,A.xy .+ B.xy,A.xz .+ B.xz,
A.yx .+ B.yx,A.yy .+ B.yy,A.yz .+ B.yz,
A.zx .+ B.zx,A.zy .+ B.zy,A.zz .+ B.zz)

-(A::AbstractTensor,B::AbstractTensor) = Tensor(A.grid,
A.xx .- B.xx, A.xy .- B.xy, A.xz .- B.xz,
A.yx .- B.yx, A.yy .- B.yy, A.yz .- B.yz,
A.zx .- B.zx, A.zy .- B.zy, A.zz .- B.zz)

dot(A::AbstractTensor,B::AbstractTensor) = Tensor(A.grid,
A.xx .* B.xx .+ A.xy .* B.yx + A.xz .* B.zx,
A.xx .* B.xy .+ A.xy .* B.yy + A.xz .* B.zy,
A.xx .* B.xz .+ A.xy .* B.yz + A.xz .* B.zz,
A.yx .* B.xx .+ A.yy .* B.yx + A.yz .* B.zx,
A.yx .* B.xy .+ A.yy .* B.yy + A.yz .* B.zy,
A.yx .* B.xz .+ A.yy .* B.yz + A.yz .* B.zz,
A.zx .* B.xx .+ A.zy .* B.yx + A.zz .* B.zx,
A.zx .* B.xy .+ A.zy .* B.yy + A.zz .* B.zy,
A.zx .* B.xz .+ A.zy .* B.yz + A.zz .* B.zz)

function ^(A::AbstractTensor,n::Integer)
  if n <= 0
    error("Only positive integers are accepted")
  elseif n == 1
    return A + zero(eltype(A))
  elseif n == 2
    return A ⋅ A
  else
    return A ⋅ (A^(n-1))
  end
end

mean(t::AbstractTensor) = SMatrix{3,3}(mean(t.xx),mean(t.xy),mean(t.xz),mean(t.yx),mean(t.yy),mean(t.yz),mean(t.zx),mean(t.zy),mean(t.zz))
std(t::AbstractTensor;mean=nothing) = mean === nothing ? SMatrix{3,3}(std(t.xx),std(t.xy),std(t.xz),std(t.yx),std(t.yy),std(t.yz),std(t.zx),std(t.zy),std(t.zz)) : SMatrix{3,3}(std(t.xx,mean=mean[1,1]),std(t.xy,mean=mean[1,2]),std(t.xz,mean=mean[1,3]),std(t.yx,mean=mean[2,1]),std(t.yy,mean=mean[2,2]),std(t.yz,mean=mean[2,3]),std(t.zx,mean=mean[3,1]),std(t.zy,mean=mean[3,2]),std(t.zz,mean=mean[3,3]))

struct Tensor{T<:AbstractFloat,N} <: AbstractTensor{T,N}
  grid::Grid
  xx::Array{T,N}
  xy::Array{T,N}
  xz::Array{T,N}
  yx::Array{T,N}
  yy::Array{T,N}
  yz::Array{T,N}
  zx::Array{T,N}
  zy::Array{T,N}
  zz::Array{T,N}
end

struct SymTensor{T<:AbstractFloat,N} <: AbstractTensor{T,N}
  grid::Grid
  xx::Array{T,N}
  yy::Array{T,N}
  zz::Array{T,N}
  xy::Array{T,N}
  xz::Array{T,N}
  yz::Array{T,N}
  yx::Array{T,N}
  zx::Array{T,N}
  zy::Array{T,N}

  function SymTensor(grid::Grid, xx::Array{T,N}, yy::Array{T,N}, zz::Array{T,N}, xy::Array{T,N}, xz::Array{T,N}, yz::Array{T,N}) where {T<:AbstractFloat,N}
    return new{T,N}(grid,xx,yy,zz,xy,xz,yz,xy,xz,yz)
  end
end

function SymTensor(f1::String,f2::String,f3::String,f4::String,f5::String,f6::String)
  nx,ny,nz,xl,yl,zl = getdimsize()
  SymTensor(Grid(xl,yl,zl),readfield(f1,nx,ny,nz),readfield(f2,nx,ny,nz),readfield(f3,nx,ny,nz),readfield(f4,nx,ny,nz),readfield(f5,nx,ny,nz),readfield(f6,nx,ny,nz))
end

SymTensor(f1::String) = SymTensor(f1,replace(f1,"11","22"),replace(f1,"11","33"),replace(f1,"11","12"),replace(f1,"11","13"),replace(f1,"11","23"))

struct ZeroArray{T,N} <: AbstractArray{T,N}
  size::NTuple{N,Int}
end
getindex(S::ZeroArray{T,N},I...) where {T,N} = zero(T)
eltype(S::ZeroArray{T,N}) where {T,N} = T
size(S::ZeroArray) = S.size

struct AntiSymTensor{T<:AbstractFloat,N} <: AbstractTensor{T,N}
  grid::Grid
  xy::Array{T,N}
  xz::Array{T,N}
  yz::Array{T,N}
  xx::ZeroArray{T,N}
  yy::ZeroArray{T,N}
  zz::ZeroArray{T,N}
  yx::MappedArray{T,N,Array{T,N},typeof(-),typeof(-)}
  zx::MappedArray{T,N,Array{T,N},typeof(-),typeof(-)}
  zy::MappedArray{T,N,Array{T,N},typeof(-),typeof(-)}

  function AntiSymTensor{T,N}(grid::Grid,xy::Array{T,N}, xz::Array{T,N}, yz::Array{T,N}) where {T,N}
    return new{T,N}(grid,xy,xz,yz,ZeroArray{T,N}(size(xy)),ZeroArray{T,N}(size(xy)),ZeroArray{T,N}(size(xy)),mappedarray((-,-),xy),mappedarray((-,-),xz),mappedarray((-,-),yz))
  end
end

function AntiSymTensor(f1::String,f2::String,f3::String)
  nx,ny,nz,xl,yl,zl = getdimsize()
  u12 = readfield(f1,nx,ny,nz)
  AntiSymTensor{eltype(u12),3}(Grid(xl,yl,zl),u12,readfield(f2,nx,ny,nz),readfield(f3,nx,ny,nz))
end

function AntiSymTensor(f1::String)
  nx,ny,nz,xl,yl,zl = getdimsize()
  u12 = readfield(f1,nx,ny,nz)
  AntiSymTensor{eltype(u12),3}(Grid(xl,yl,zl),u12,readfield(replace(f1,"12","13"),nx,ny,nz),readfield(replace(f1,"12","23"),nx,ny,nz))
end

# =============================== Mixed Functions ==========================================

dot(t::AbstractTensor,u::FVector{T,N}) where {T,N} = FVector(u.grid,
t.xx .* u.x .+ t.xy .* u.y + t.xz .* u.z,
t.yx .* u.x .+ t.yy .* u.y + t.yz .* u.z,
t.zx .* u.x .+ t.zy .* u.y + t.zz .* u.z)

dot(u::FVector{T,N},t::AbstractTensor) where {T,N} = FVector(u.grid,
t.xx .* u.x .+ t.yx .* u.y + t.zx .* u.z,
t.xy .* u.x .+ t.yy .* u.y + t.zy .* u.z,
t.xz .* u.x .+ t.yz .* u.y + t.zz .* u.z)

function ∇(field::Scalar)
   FVector(field.grid,dx(field).data,dy(field).data,dz(field).data)
end
function ∇(field::FVector)
  Tensor(field.grid,
  dx(field.x,field.grid.xl),dx(field.y,field.grid.xl),dx(field.z,field.grid.xl),
  dy(field.x,field.grid.yl),dy(field.y,field.grid.yl),dy(field.z,field.grid.yl),
  dz(field.x,field.grid.zl),dz(field.y,field.grid.zl),dz(field.z,field.grid.zl),)
end

function ∇²(field::Scalar)
  Scalar(field.grid,
  dx(field.data,field.grid.xl,2) .+ dy(field.data,field.grid.yl,2) .+ dz(field.data,field.grid.zl,2))
end

function ∇²(field::FVector)
  FVector(field.grid,
  dx(field.x,field.grid.xl,2) .+ dy(field.x,field.grid.yl,2) .+ dz(field.x,field.grid.zl,2),
  dx(field.y,field.grid.xl,2) .+ dy(field.y,field.grid.yl,2) .+ dz(field.y,field.grid.zl,2),
  dx(field.z,field.grid.xl,2) .+ dy(field.z,field.grid.yl,2) .+ dz(field.z,field.grid.zl,2))
end

dot(v::FVector,u::FVector) = Scalar(v.grid, v.x .* u.x .+ v.y .* u.y .+ v.z .* u.z)

+(v::SymTensor,u::AntiSymTensor) = Tensor(v.grid,   v.xx     , v.xy .+ u.xy, v.xz .+ u.xz,
                                                 v.xy .- u.xy,     v.yy    , v.yz .+ u.yz,
                                                 v.xz .- u.xz, v.yz .- u.yz,     v.zz)
+(u::AntiSymTensor, v::SymTensor) = v+u


*(s::AbstractScalar,v::FVector) = FVector(v.grid,s .* v.x, s .* v.y, s .* v.z)

*(u::FVector, s::AbstractScalar) = *(s,u)

/(v::FVector,s::AbstractScalar) = FVector(v.grid, v.x ./ s, v.y ./ s, v.z ./ s)

dot(f::typeof(∇),u::FVector) = Scalar(u.grid, dx(u.x,u.grid.xl) .+ dy(u.y,u.grid.yl) .+ dz(u.z,u.grid.zl))

dot(f::typeof(∇),u::AbstractTensor) = FVector(u.grid,
dx(u.xx,u.grid.xl) .+ dy(u.yx,u.grid.yl) .+ dz(u.zx,u.grid.zl),
dx(u.xy,u.grid.xl) .+ dy(u.yy,u.grid.yl) .+ dz(u.zy,u.grid.zl),
dx(u.xz,u.grid.xl) .+ dy(u.yz,u.grid.yl) .+ dz(u.zz,u.grid.zl))

Base.cross(f::typeof(∇),u::FVector) = FVector(u.grid,
dy(u.z,u.grid.yl) .- dz(u.y,u.grid.zl),
dz(u.x,u.grid.zl) .- dx(u.z,u.grid.xl),
dx(u.y,u.grid.xl) .- dy(u.x,u.grid.yl))

mod(v::FVector) = Scalar(v.grid, sqrt.(v.x .^2 .+ v.y .^2 .+ v.z .^2))
mod(v::AbstractTensor) = Scalar(v.grid, @.(sqrt(v.xx^2 + v.xy^2 + v.xz^2 + v.yx^2 + v.yy^2 + v.yz^2 + v.zx^2 + v.zy^2 + v.zz^2)))

end # module
