# Definition for the coefficients found in the sh_to_rgb function

Certainly! Here's an expanded table that includes spherical harmonics up to \( l = 3 \) in Cartesian coordinates. The expressions grow more complex with higher values of \( l \) and \( m \).

| \( l \) | \( m \) | Spherical Harmonic \( Y_{l}^{m} \) in Cartesian Coordinates |
|---------|---------|------------------------------------------------------------|
| 0       | 0       | \( Y_{0}^{0} = \frac{1}{2\sqrt{\pi}} \) |
| 1       | -1      | \( Y_{1}^{-1} = \frac{\sqrt{3}}{2\sqrt{\pi}} \frac{y}{r} \) |
| 1       | 0       | \( Y_{1}^{0} = \frac{\sqrt{3}}{2\sqrt{\pi}} \frac{z}{r} \) |
| 1       | 1       | \( Y_{1}^{1} = \frac{\sqrt{3}}{2\sqrt{\pi}} \frac{x}{r} \) |
| 2       | -2      | \( Y_{2}^{-2} = \frac{\sqrt{15}}{2\sqrt{\pi}} \frac{xy}{r^2} \) |
| 2       | -1      | \( Y_{2}^{-1} = \frac{\sqrt{15}}{2\sqrt{\pi}} \frac{yz}{r^2} \) |
| 2       | 0       | \( Y_{2}^{0} = \frac{\sqrt{5}}{4\sqrt{\pi}} \frac{2z^2 - x^2 - y^2}{r^2} \) |
| 2       | 1       | \( Y_{2}^{1} = \frac{\sqrt{15}}{2\sqrt{\pi}} \frac{xz}{r^2} \) |
| 2       | 2       | \( Y_{2}^{2} = \frac{\sqrt{15}}{4\sqrt{\pi}} \frac{x^2 - y^2}{r^2} \) |
| 3       | -3      | \( Y_{3}^{-3} = \frac{\sqrt{35}}{4\sqrt{\pi}} \frac{y (3x^2 - y^2)}{r^3} \) |
| 3       | -2      | \( Y_{3}^{-2} = \frac{\sqrt{105}}{2\sqrt{\pi}} \frac{xy z}{r^3} \) |
| 3       | -1      | \( Y_{3}^{-1} = \frac{\sqrt{21}}{4\sqrt{\pi}} \frac{y (4z^2 - x^2 - y^2)}{r^3} \) |
| 3       | 0       | \( Y_{3}^{0} = \frac{\sqrt{7}}{4\sqrt{\pi}} \frac{2z^3 - 3z (x^2 + y^2)}{r^3} \) |
| 3       | 1       | \( Y_{3}^{1} = \frac{\sqrt{21}}{4\sqrt{\pi}} \frac{x (4z^2 - x^2 - y^2)}{r^3} \) |
| 3       | 2       | \( Y_{3}^{2} = \frac{\sqrt{105}}{4\sqrt{\pi}} \frac{z (x^2 - y^2)}{r^3} \) |
| 3       | 3       | \( Y_{3}^{3} = \frac{\sqrt{35}}{4\sqrt{\pi}} \frac{x (x^2 - 3y^2)}{r^3} \) |

This table provides the expressions for spherical harmonics in Cartesian coordinates for \( l = 0 \) to \( l = 3 \). The expressions are derived by substituting the spherical coordinates with their Cartesian equivalents. As \( l \) and \( m \) increase, the expressions involve higher powers of \( x \), \( y \), and \( z \), and become more complex.