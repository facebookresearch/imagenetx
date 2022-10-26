export function* chunkArray(arr, n) {
  for (let i = 0; i < arr.length; i += n) {
    yield arr.slice(i, i + n);
  }
}

export function zipArray(arrays) {
  let z = [];
  for (let i = 0, len = arrays[0].length; i < len; i++) {
    let p = [];
    for (let k = 0; k < arrays.length; k++) {
      p.push(arrays[k][i]);
    }
    z.push(p);
  }
  return z;
}
