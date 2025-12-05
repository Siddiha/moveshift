# Import Three.js library and type definition
import * as THREE from 'three'
import type { SerializableGeometry } from '../types'

# Function to convert Three.js BufferGeometry to serializable format
export function serializeGeometry(geometry: THREE.BufferGeometry): SerializableGeometry {
  # Get position attribute from geometry (x, y, z coordinates)
  const pos = geometry.getAttribute('position') as THREE.BufferAttribute | null
  # Get normal attribute from geometry (surface direction vectors)
  const norm = geometry.getAttribute('normal') as THREE.BufferAttribute | null
  # Get UV attribute from geometry (texture coordinates)
  const uv = geometry.getAttribute('uv') as THREE.BufferAttribute | null
  # Get index buffer (defines triangle faces)
  const index = geometry.getIndex()
  
  # Return serialized geometry data structure
  return {
    # Convert position data to regular JavaScript array
    positions: pos ? Array.from(pos.array as Float32Array) : [],
    # Convert normal data to regular JavaScript array (optional)
    normals: norm ? Array.from(norm.array as Float32Array) : undefined,
    # Convert UV data to regular JavaScript array (optional)
    uvs: uv ? Array.from(uv.array as Float32Array) : undefined,
    # Convert index data to regular JavaScript array (optional)
    indices: index ? Array.from(index.array as Uint16Array | Uint32Array) : undefined,
  }
}

# Function to reconstruct Three.js BufferGeometry from serialized data
export function deserializeGeometry(data: SerializableGeometry): THREE.BufferGeometry {
  # Create new empty BufferGeometry
  const geometry = new THREE.BufferGeometry()
  
  # Reconstruct position attribute if data exists
  if (data.positions && data.positions.length > 0) {
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(data.positions), 3))
  }
  
  # Reconstruct normal attribute if data exists
  if (data.normals && data.normals.length > 0) {
    geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(data.normals), 3))
  }
  
  # Reconstruct UV attribute if data exists
  if (data.uvs && data.uvs.length > 0) {
    geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(data.uvs), 2))
  }
  
  # Reconstruct index buffer if data exists
  if (data.indices && data.indices.length > 0) {
    # Check if we need 32-bit indices (if any index > 65535)
    const needsUint32 = Math.max(...data.indices) > 65535
    # Create appropriate index buffer based on maximum index value
    geometry.setIndex(new THREE.BufferAttribute(
      needsUint32 ? new Uint32Array(data.indices) : new Uint16Array(data.indices), 
      1
    ))
  }
  
  # Compute bounding box for geometry (useful for frustum culling)
  geometry.computeBoundingBox()
  # Compute bounding sphere for geometry (useful for distance calculations)
  geometry.computeBoundingSphere()
  
  # Compute vertex normals if they don't already exist
  # (necessary for proper lighting)
  if (!geometry.getAttribute('normal')) geometry.computeVertexNormals()
  
  # Return the fully reconstructed geometry
  return geometry
}
