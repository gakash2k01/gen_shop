export const percentOffCalc = (price, originalPrice) => {
    const discount = (originalPrice - price)*100/originalPrice;
    return Math.round(discount);
}