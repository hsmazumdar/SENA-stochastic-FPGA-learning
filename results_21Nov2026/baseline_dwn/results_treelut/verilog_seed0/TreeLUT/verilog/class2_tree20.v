module class2_tree20(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if((i[47] & i[422] & (~i[436]) & i[350]) | ((~i[47]) & i[422] & (~i[436]) & i[350]) | ((~i[212]) & i[245] & i[181] & i[436] & i[350]) | (i[229] & (~i[245]) & i[181] & i[436] & i[350]) | (i[203] & i[184] & (~i[181]) & i[436] & i[350]) | ((~i[203]) & i[184] & (~i[181]) & i[436] & i[350]) | ((~i[233]) & (~i[184]) & (~i[181]) & i[436] & i[350]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
