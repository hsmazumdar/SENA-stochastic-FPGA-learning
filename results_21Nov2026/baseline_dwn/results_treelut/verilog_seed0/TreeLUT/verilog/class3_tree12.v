module class3_tree12(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[162]) & (~i[140]) & i[121] & i[447] & i[36]) | (i[344] & i[103] & (~i[121]) & i[447] & i[36]) | ((~i[344]) & i[103] & (~i[121]) & i[447] & i[36]) | ((~i[318]) & (~i[103]) & (~i[121]) & i[447] & i[36]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
